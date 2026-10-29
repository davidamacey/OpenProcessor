# Cluster configuration decision after the 3,000-item baseline

Status: decision note, 2026-10-29. No product code changed. Written on branch
`docs/cluster-config-decision` from `feat/cluster-w0b-baseline` at `619a3012`. Inputs:
[cluster_config_baseline.md](cluster_config_baseline.md) (E1, E5 and diagnostics on the 3,000-item
set), [cluster_eval_baseline.md](cluster_eval_baseline.md) (production baselines A and B),
[`cluster_config_baseline_3000.json`](../benchmarks/cluster_config_baseline_3000.json), the
consolidated research (`git show docs/cluster-methods-review:docs/research/clustering_research_consolidated.md`),
the program plan (`git show docs/cluster-search-plan:docs/design/cluster_search_makemodel_plan.md`),
the raw W0b outputs (`artifacts_local/cluster_search/w0b/e1_3000/`, gitignored) and the clustering
code under `src/services/curation/clustering/`.

Hardware for the new checks in this note: CPU only, shared host, 1-minute load average 50 to 60, so
nothing here is a timing. The scorer is deterministic.

Naming: the consolidated note numbers the kinds K1 whole image, K2 crops, K3 faces, K4
fine-grained; the plan numbers K3 region boxes and K4 faces. This note uses names, not numbers.

## 1. Decision in one paragraph

Change the residual-pool partition from global IVF cells (K = min(512, n // 6)) to **average-linkage
AHC inside each detector class**: the partition key is the normalized `detector_class_name`, one
cosine-distance cut per embedding model and project, chosen on a calibration split. Keep IVF only as
an index. Add no automatic cross-class merge stage. Instead, give each within-class cluster an
explicit **novel-candidate state** for review, set when it is far from the class's own validated
examples. Do not implement it yet. First run the cheap experiments in section 6 on the 3,000-item
set, and the set P cut-transfer check once the judge has passed its agreement run (E0). Then build
it in its own wave (section 8). The evidence is one public set and one held-out half. It also leans
on a detector trained on the same label space as most of that set. Section 9 lists what that does
not prove.

## 2. What the evidence says, including three checks added here

The W0b result as reported (held-out half, cut chosen on the calibration half):

- IVF now: pair precision 0.734, recall 0.029.
- Global average linkage: 0.40 at 1.7 times that recall and 0.26 at 9 times. It never reaches IVF's
  precision at a higher recall.
- Leiden CPM: beats global average linkage only at recall 0.03 to 0.07 (0.64 to 0.79). It never
  reaches recall 0.1, leaves 4 to 11 % of crops unassigned, and its ARI across seeds and row orders
  is 0.64 to 0.69.
- Within the detector class, on the low-confidence subsets: held-out precision 0.956 to 0.969,
  against 0.14 to 0.23 for the global cut.

Reading the raw data for this decision turned up three things the W0b note does not show. All three
use the same bundle and split. Scripts and logs are in
`artifacts_local/cluster_search/w0b/decision_checks/` (gitignored; see Reproduce).

**(a) The detector-class agreement figure is understated by a name-format mismatch.** `run.py` (E5
and the diagnostics' label facts) compares `detector_class_name` (registry slug form, for example
`traffic_light`) with the ground-truth label in COCO form (`traffic light`) as raw strings. The
product's one name-equality rule (`normalize_class_name` in `src/utils/class_names.py`) treats them
as the same class. With that rule:

| Subset | Agreement as reported | Agreement, names normalized | On labels the detector can name |
|---|---|---|---|
| all labelled crops (2,950) | 0.755 | **0.820** | 0.954 |
| confidence < 0.7 (1,282) | 0.729 | **0.810** | 0.932 |
| confidence < 0.5 (707) | 0.724 | **0.812** | 0.927 |

The real disagreement is about 18 %, not 27 %, and it has two parts:

- **Open set (78 % of it).** 413 of the 530 disagreeing crops are the 12 Open Images animals the
  COCO detector has no class for (alpaca, camel, fox, goat, kangaroo, lion, owl, penguin, pig,
  rabbit, squirrel, tiger). The detector files them under bird (85), sheep (84), bear (78), dog (66),
  cat (48), cow (21) and horse (20).
- **Closed-set confusion (the other 117).** Mainly truck and car (32), cow and sheep (13), and
  motorcycle and person (9).

The partitions are unaffected (they group by the slug consistently). Only the reported agreement
rate and the "73 %" that this task quotes are off. Fixing the comparison is item X0 in section 6.

**(b) Pair metrics on this set mostly measure the person class, and the detector alone already
scores high on them.**

- Person crops are 28 % of the labelled crops but 78 to 81 % of the same-label pairs in each scored
  subset.
- A partition that is nothing but the detector class, with no clustering at all (69 buckets), scores
  held-out pair precision / recall **0.906 / 0.960**, and BCubed precision / recall / F **0.786 /
  0.870 / 0.826**. That BCubed precision equals IVF's (0.792), and its recall is 4.7 times IVF's.
- The E5 within-class row at target recall 0.5 (cut 0.593, held out 0.901 / 0.859) is effectively
  that bucket partition.

So E5's 0.96 pair precision is mostly the detector's doing. What the clustering itself adds shows
on BCubed and on the non-person crops:

| Held-out half (1,494 labelled; 1,084 non-person) | Pair P / R | BCubed P / R / F | Non-person BCubed P / R |
|---|---|---|---|
| IVF now (production configuration, refit) | 0.734 / 0.029 | 0.792 / 0.187 / 0.302 | 0.814 / 0.252 |
| Detector class only (no clustering) | 0.906 / 0.960 | 0.786 / 0.870 / 0.826 | 0.711 / 0.828 |
| Global average linkage, cut 0.307 | 0.399 / 0.073 | 0.741 / 0.285 / 0.412 | 0.762 / 0.381 |
| **Within detector class, cut 0.346** (chosen on calibration at target 0.05 and 0.1) | **0.924 / 0.129** | **0.943 / 0.357 / 0.518** | **0.925 / 0.461** |
| **Within detector class, cut 0.381** (chosen at target 0.2) | **0.919 / 0.241** | **0.920 / 0.457 / 0.611** | **0.895 / 0.561** |
| Within detector class, cut 0.425 (target 0.3) | 0.906 / 0.312 | not scored | not scored |

Reading the table:

- **Fold 2 agrees.** Cut chosen on the held-out half and scored on calibration: 0.945 / 0.181 at
  cut 0.361, and 0.909 / 0.304 at cut 0.425. The chosen cuts stay between 0.35 and 0.43 at every
  target up to 0.3. The committed `run.py e5 --confidence-cuts 1.01` reproduces every
  within-class pair cell above.
- **The gaps hold across subsamples.** Over 300 random halves of the held-out images, BCubed
  precision ranges 0.805 to 0.864 for IVF, against 0.931 to 0.971 and 0.908 to 0.955 for the two
  within-class cuts. Pair precision ranges 0.697 to 0.811 for IVF, against 0.868 to 0.968 and 0.840
  to 0.962. These are spreads, not confidence intervals: BCubed on a smaller set reads high. They
  show only that the gaps do not close under resampling.
- **Per stratum** (held-out BCubed precision, IVF against cut 0.346): low resolution 0.803 against
  0.968 (recall 0.209 against 0.606), mixed scene 0.835 against 0.978, near-duplicate 0.935 against
  0.972, rare 0.956 against 0.986, similar classes 0.888 against 0.934, elongated more than 2:1 0.895
  against 0.978. No stratum loses precision.
- **The clustering's own contribution** is splitting the detector's buckets. On non-person crops,
  BCubed precision rises from 0.711 (buckets alone) to 0.925, while keeping 1.8 times IVF's recall.

**(c) The open-set crops behave differently from the closed-set ones, and confidence cannot find
them.**

- **Hard-partition cost.** A hard partition splits only 3.1 % of the same-label held-out pairs among
  labels the detector can name, but 48.2 % among the open-set labels (4.0 % overall).
- **Purity.** Inside their (wrong) detector bucket, open-set crops still land in fairly pure
  sub-clusters at cut 0.381. On average 0.79 of a crop's own cluster shares its label, against 0.94
  for nameable crops. 4.3 % of open-set crops end up as singletons, against 7.8 % of nameable crops.
- **Confidence.** The detector is more confident on the open-set crops (median 0.801) than on its
  correct calls (0.759). Its genuinely confused closed-set calls sit at 0.589. As a score for "this
  is an unknown class", 1 minus the detector confidence has AUROC 0.638.
- **Novelty against validated examples.** A score built from same-class examples ranks the open-set
  crops well: the mean cosine to the 10 nearest calibration-half crops whose label equals the
  detector class, standing in for validated examples. Per detector class the AUROC is bird 0.986,
  cat 1.000, cow 0.941, dog 0.998, horse 0.995 and sheep 0.919. Pooled across classes it is 0.852,
  because the score is not calibrated across classes. These samples are small: bird has 3 own-class
  query crops, cow 44. Read this as a direction only.

## 3. Question 1: the production design change, and what it must not do

### The change (for a later wave; not implemented here)

1. **Partition key.** The item's `detector_class_name`, normalized with `normalize_class_name`. Never
   `detector_class_id`: class identity is the name. An item with no detector class gets the explicit
   key `none`. It is clustered within that key, never dropped and never mixed into a class bucket.
2. **Method inside each key.** Average-linkage AHC with one cosine-distance cut. The input is sorted
   by document id (as #372 now does), so the run is a function of the document set (permutation ARI
   1.000 measured for fixed-cut average linkage).
3. **The cut** is stored with the run record, together with the embedding model and field
   (`OP_RESIDUAL_EMBEDDING_FIELD`) and its source:
   - `default`: the per-model value calibrated on the public set. For PE-Core on this set the
     calibrated values were 0.35 to 0.38 at operating recall up to 0.2.
   - `project`: calibrated on the project's own validated items, when it has enough of them. The
     minimum counts are set by experiment X5.

   The set P projects on the dev stack have 0 validated items (W0 inventory), so `default` is the
   only source there today. Whether it transfers is open (X5, X8).
4. **IVF stays as an index.** It stays for streaming assignment and kNN candidates only. Ingest-time
   assignment must look only at clusters of the item's own partition key, and must abstain when the
   nearest is beyond the cut or the margin is small (X6). A nearest centroid across classes would
   undo the partition.
5. **Novel-candidate state, not an "unknown" bucket.** A within-class cluster far from its class's
   validated examples is flagged for review. The flag is a cluster state, not a shared cluster id:
   pooling all unknowns would merge penguins with owls, which today sit in separate, fairly pure
   sub-clusters.
6. **Cross-class links only as suggestions.** A second-stage merge is suggested only between
   novel-candidate clusters from different detector buckets (for example alpaca under sheep and
   alpaca under dog). A person or the loop gate decides. It ships only if X4 shows its suggestions
   are precise.
7. **Scale.** Dense AHC on a 347k pool took about an hour and 124 GB
   ([clustering_methods.md](clustering_methods.md) 2.2). Partitioning lowers n per bucket, but the
   largest bucket (person or car) decides the cost. X7 measures that bucket on the 91,184-item set P
   project. Above the measured limit, the plan's sparse graph-HAC route applies within the bucket.

### What it must not do

- **No global cut over all classes.** At 86 classes, global average and complete linkage collapse
  (held-out precision 0.16 to 0.40 at every target from recall 0.05 up), and Leiden cannot reach
  recall 0.1.
- **Never write the detector class as the item's class**, and never treat a within-class cluster's
  majority as confirming the detector. With the partition keyed on the detector class, "cluster
  majority agrees with the detector" holds by construction. Any rule that counts it as independent
  evidence becomes circular: `auto_promote.py` (`cluster_majority_agreement`), or a loop gate fed by
  detector-derived names.

  The concrete risk is open-set crops: 0.79-pure clusters of penguins inside "bird" would be stamped
  "bird". Auto-promote stays disabled (its default). Class writes stay behind the loop's sampled gate
  with an independent labeler (VLM or person) and the lock rule.
- **Do not use detector confidence to find unknowns** (AUROC 0.638; open-set crops are the more
  confident ones).
- **No Leiden, ECG or 2b k-means** as the residual partition (section 5), and no whitening in the
  product before WB.
- **No calibration on the held-out half, and no claim beyond this set** until X5, X8 and X9 pass.

## 4. Question 2: the crops where the detector class disagrees with the label

**Size.** About 18 % of labelled crops, not 27 % (finding (a)). On labels the detector can name it is
5 to 7 %. The rest are classes the detector cannot output.

**Soft prior or hard partition: hard for grouping, soft for decisions.**

- **Why hard for grouping.** The hard partition is what produced the precision. It costs 3.1 % of the
  same-label pairs among nameable labels. A soft version (a distance penalty for crossing classes)
  adds a second calibrated parameter with no evidence that it beats the hard key plus the suggestion
  stage. The cost is concentrated in the open-set labels (48 % of their pairs split). Those are
  exactly the crops the novel-candidate state and the cross-class suggestions target.
- **What soft for decisions means.**
  - The detector class is never a label, never locks anything, and never vetoes a human or loop
    move across classes.
  - A person can move an item to any class.
  - Cross-class merges exist only as suggestions.
  - The plan's veto across coarse classes "only when both are confident" is not supported here.
    Confidence does not separate right from wrong calls on open-set objects (median 0.80 for wrong
    calls against 0.76 for right ones). Do not add a veto.

**Fail-closed behaviour:**

- **Missing detector class.** It goes to the explicit `none` key. The key is recorded in the run, and
  the count is reported.
- **No calibrated cut** for the run's embedding model or field: refuse the run, with a typed error
  naming the field. Never fall back to another model's cut. A cut is meaningful only in its own space:
  set P's mean random-pair cosine is 0.63, against 0.46 to 0.50 on public crops.
- **A project whose detector cannot split the pool.** Examples: a single-class or class-agnostic
  detector, or a pool where one key holds most items across many true classes. Within-class then
  degenerates into the global cut that collapsed here. The run must report the largest-key share. A
  threshold for warning or refusing comes from X5 and X7, not from this note.
- **Mixed detectors.** Projects whose items come from several detector profiles stay keyed by name
  (by-name rule). The run record lists the profiles present.

**Lock rule and the stores:**

- The residual pool already excludes validated, confidently sourced and excluded items
  (`residual_fetch.py`, `residual_gate.py`).
- `cluster_write_guard.py` allows cluster-id writes on unvalidated holdout items, because a cluster
  id is not a class. Nothing in this change writes `class_name`, so the lock rule is untouched.
- Writes stay on the existing guarded bulk path.
- Region boxes are a separate pool: regions of one profile are one class
  (`region_box_clustering.py`), so the detector-class key does not apply to them. They are not
  changed by this decision (question 5).

## 5. Question 3: the kill rules, arm by arm

"Confirmed" means the kill rule (or adoption rule) of the consolidated note or plan holds on the
3,000-item held-out half. "Killed" means the rule fired against the arm.

| Arm | Rule (source) | Result on the 3,000-item set | Verdict |
|---|---|---|---|
| Average-linkage AHC, global | Candidate default; adopt as the residual partition "only if E1 confirms" (plan W0b) | Held out 0.40 at recall 0.073 and 0.26 at 0.26, against IVF 0.734 at 0.029; no cut matches IVF at higher recall | **Global default killed.** The 11-class COCO sample misled |
| Average-linkage AHC within the detector class | E5 adoption: precision rises at equal recall on the residual-like subset | Held out 0.956 to 0.969 against 0.14 to 0.23 (low-confidence subsets); all confidences, cut 0.346, BCubed P 0.943 against 0.792 at 1.9 times IVF's BCubed recall; fold 2 agrees | **Confirmed**, with the person-dominance and detector caveats of section 2(b) |
| Complete-linkage AHC | Dominated by average linkage at class level (consolidated note 4.2) | Below average linkage in every oracle cell; held out 0.395 / 0.075 at best; no cut reaches recall 0.3 | **Kill confirmed** for crops. The identity-level question (faces) stays open in E6 |
| Leiden CPM, mutual 30-NN | Not default unless it beats average linkage at the chosen recall | Beats global average linkage only at recall under 0.07 (0.64 to 0.79), never reaches 0.1, 4 to 11 % unassigned, ARI 0.64 to 0.69 under seed or row order (fails plan principle 9). Within-class Leiden not run | **Killed as default.** Within-class Leiden is not scheduled: its instability fails principle 9 whatever its precision, and within-class average linkage is already at 0.92 to 0.95 |
| ECG, once | Run once on real data (consolidated E1) | 0.376 / 0.200, 26 clusters, best AMI (0.547) but pair precision far below IVF; ARI 0.641 | **Killed.** The synthetic 0.953 / 0.899 does not transfer |
| HDBSCAN | Rejected on real data (COCO sample, about 350k crops) | Not re-run; nothing in the new data argues for density gaps in PE-Core crop space | **Kill stands** for crops. Core-point rules stay valid for faces (Immich-style) |
| FINCH | Rejected on the COCO sample | Not re-run | **Kill stands** |
| PCA whitening 256 | Kill if held-out precision gain at matched recall is under 0.02 | P@10 0.686 against 0.679 raw; oracle curves at or below raw at every recall (global average linkage). Held-out within-class not run | **Killed for the global representation.** One within-class held-out check (X3) closes it; ship only through WB's rule |
| k-reciprocal re-ranking | High-recall arm, test with Leiden on the 3,000 set (consolidated table 6) | P@10 +0.012; hub skew 2.78 to 0.75; oracle at recall 0.1 and 0.2: 0.425 and 0.367, against raw 0.331 and 0.253; at 0.3 below raw (0.145 against 0.198); none at 0.5 | **Survives, unproven.** The lift is on the global frontier, which is no longer the design; batch only (no incremental form). Test within class held out (X3) |
| DBA / alpha-QE | Drop (consolidated: no clustering gain on COCO) | Oracle 0.402 at recall 0.1 (raw 0.331), below raw at 0.2 (0.215 against 0.253) | **Drop decision contradicted at recall 0.1 only.** Include in X3 at no extra cost; drop unless held out within class it gains 0.02 |
| CSLS, centring | Optional / drop | At or below raw in every oracle cell; P@10 +0.010 | **Killed** |
| 2b MiniBatchKMeans, k = max(8, round(n / 800)) | Score it (consolidated E1) | Held out 0.187 / 0.273 on items, 8 clusters, ARI 0.553 | **Killed as a cluster set** (it is a bucket index). Its real job, region boxes of one class, is not what E1 scored: replacing it needs a box-level set (question 5) |
| IVF, K = n // 6 | Keep as index, stop treating cells as clusters | Held out 0.734 / 0.029; BCubed recall 0.187; ARI 1.000 after the #372 sort | **Confirmed**: index only |

## 6. Question 4: the cheapest next experiments, with acceptance bars

All of these run on CPU on the stored 3,000-item vectors unless marked. Each is minutes to about an
hour. Rules for every one:

- The cut and every parameter are chosen on `calibration`, scored on `heldout`, and both folds are
  reported.
- Every table carries the detector-only row and the non-person columns, beside pair and BCubed.
- Scorecards come from `scripts/bench/cluster_eval.py`, with three identical repetitions.

| ID | Experiment | Acceptance bar (all on held out, both folds unless stated) | Cost |
|---|---|---|---|
| X0 | Compare names with `normalize_class_name` in `run.py` (E5 agreement, diagnostics label facts) and add a hand-computed test; re-state the 3,000-set agreement in [cluster_config_baseline.md](cluster_config_baseline.md) | Agreement reproduces 0.820 / 0.810 / 0.812 | minutes |
| X1 | Land the within-class arm in the committed scripts: an `export_partition.py` arm for within-detector-class average linkage, plus a detector-only arm, scored with `cluster_eval.py` (BCubed, AMI, strata, non-person block). Replaces the scratch checks of section 2 | Reproduces section 2 to three decimals | under 1 h |
| X2 | Product bar, on the X1 output, at the cut chosen for target recall 0.1 | All of: BCubed precision >= 0.90 overall, >= 0.88 on non-person and >= 0.85 on low resolution; BCubed recall >= 2 x IVF's (>= 0.37); pair recall >= 3 x 0.022; no stratum's BCubed precision below IVF's; permutation ARI 1.000; 0 protected items moved. The section-2 cuts pass the precision, stratum and pair-recall bars; cut 0.346 misses the BCubed-recall bar (0.357 against 0.37), and permutation ARI and protected moves are not yet measured for the within-class arm | minutes |
| X3 | Representation arms inside the detector class: raw, PCA-whitened 256 (fit on the independent set P pool), k-reciprocal (k1 20, k2 6, lambda 0.3), DBA | An arm passes only if BCubed precision at matched BCubed recall (within 0.02) rises by >= 0.02 in both folds and no stratum loses more than 0.02. Otherwise it is closed for crops and WB does not re-test it | about 1 h |
| X4 | Novelty and cross-class suggestions: score each within-class cluster by its distance to the class's validated examples (calibration-half crops whose label equals the detector class); suggest a merge between novel clusters of different keys by best member-pair cosine | Novelty: per-class AUROC >= 0.90 on every key with >= 20 examples and >= 10 novel crops, and flag precision >= 0.80 at the threshold chosen on calibration. Suggestions: precision of the top suggestions >= 0.90, and pair recall among open-set labels improves without lowering overall BCubed precision by more than 0.01. Below either bar the matching state or stage is not built | about 1 h |
| X5 | Cut transfer: apply the per-model cut chosen on the 3,000 calibration half, unchanged, to the 200-image COCO project's held-out half (`splits/coco_sample_200.json`); also find how many validated items per key a project-level calibration needs (subsample the calibration half to 5, 10, 20 and 50 per key) | COCO held out BCubed precision >= 0.90 at the transferred cut; the minimum per-key count is the smallest whose chosen cut stays within 0.02 of the full-calibration cut in 9 of 10 seeded subsamples | under 1 h |
| X6 | Streaming assignment: treat 20 % of the images (seeded, by image) as late arrivals; assign each to the nearest cluster of its own key (centroid, and best member) with abstention beyond the cut or below a margin; compare with a full refit | Accepted assignments' label precision >= 0.95; abstain share reported; ARI of incremental against full refit reported (no bar yet) | minutes |
| X7 | Scale on real vectors: per-key counts of the 91,184-item set P project (read-only export, numbers only); time and peak RSS of dense within-key average linkage for the largest key, and of connectivity-constrained average linkage on a mutual 30-NN graph | Dense within-key AHC is the product method only if the largest key fits in 30 min and 32 GB on one host; otherwise the sparse variant is used, and it must reproduce the 3,000-set held-out row within 0.01 BCubed | 1 to 2 h |
| X8 | Set P, numbers only, after W8's E0 has passed (pair kappa >= 0.7, flag precision >= 0.8) and with an owner-approved sample: on the same seeded items, contact sheets of their cluster under IVF now and under the within-class default cut. Local VLM on up to 40 cluster sheets per arm; Haiku on at most 20 per wave (O3) | Judge purity of within-class clusters not below IVF's, with a Wilson 95 % lower bound >= 0.75 on >= 35 sheets; largest cluster <= 10 % of its key unless the judge rates it pure; a run below 35 sheets is reported as "insufficient sample", not a pass | judge budget only |
| X9 | Split spread: three more seeded splits of the 3,000-item pin by source image (the pinned split stays the acceptance split) | The X2 bars hold on the pinned split; the spread across the four splits is reported beside them. If any split fails a bar by more than 0.03, the decision is reopened | under 1 h |

Order: X0 and X1 first (they make the evidence reproducible from committed code), then X2, X3, X4,
X5, X6 and X9 in any order, X7 when the set P export is available, and X8 after W8 E0.

Results of the CPU experiments (2026-10-29), each against its bar, in
[cluster_config_experiments.md](cluster_config_experiments.md): X0 pass, X1 pass (three non-person cells of the
section 2 table differ by up to 0.003, see that note), X2 **fail** (BCubed recall 0.357 against 0.37; every
precision bar passes), X3 **fail** for all three arms, X4 **fail**, X5 transfer pass and sample size **fail**
as specified, X6 **fail** (marginal), X7 dense method fits (172 s, 12.7 GiB for the 40,016-crop person key) and
the sparse variant fails equivalence, X9 reopen rule fires on fold 2 only. X8 is not run. The verdicts above and
the plan in section 8 are unchanged by this entry.

## 7. Question 5: faces and whole images

**What transfers:**

- **Protocol.** Calibration-chosen cut, held-out score, both folds. Large-class dominance must be
  checked: in pair metrics the largest class can carry 80 % of the positive pairs. Report BCubed and
  a free-prior or trivial baseline row next to any method row, because the detector alone matched
  IVF's BCubed precision here.
- **Diagnosis.** A single global cut collapses when many groups share a space with weak density
  gaps. The fix comes from structure the pipeline already has, not from a cleverer global algorithm.
- **Engineering rules.** Order independence by id sort, cuts stored per model and field, and no
  shared "unknown" bucket.

**What does not transfer, and must be measured separately:**

- **Faces.** Identity level, with ArcFace, a space trained for identity with real density gaps. There
  is no class prior: every face is "face". The analogous free signals are the same-image cannot-link
  and named persons as anchors, not a partition key. The method question (union-find at 0.7 against
  Immich-style core points at 0.5 / 3, average or complete graph HAC, Leiden) is E6 on LFW and a pose
  set, held out, as planned. Nothing in this note changes E6's bars. Complete linkage, killed here
  for crops, is still a live arm for identity. Faces never go to a hosted judge.
- **Whole images** (MobileCLIP 512, scene or topic level). No detector class exists per image. A
  possible analogue is the set of detected classes, or a scene tag, as a soft key. It is unmeasured,
  and it is the only new arm this result suggests for E7. E7 otherwise stays as planned (hierarchical
  k-means against flat IVF against average linkage, label-set Jaccard on COCO and Open Images).
- **Region boxes (2b).** One class per profile, so "within class" is the whole pool. What might
  transfer is "average linkage with a calibrated cut instead of k-means at n / 800". That needs the
  box-level set of the plan's region-box row. E1's item numbers do not decide it.
- **Fine-grained groups** (make/model). This result supports the plan's structure: a fine graph per
  coarse class. It says nothing about whether PE-Core separates sub-classes inside a class. That is
  E9 on Stanford Cars or CompCars, held out and class-disjoint.

## 8. Question 6: changes to the plan's waves

1. **Close W0b as research only.** Its clause "product default (only if E1 confirms): sparse kNN
   average-linkage graph HAC" did not hold as written (the global form failed). W0b gives no product
   default.
2. **New research wave W0c (CPU, before any implementation).** Runs X0 to X7 and X9 of section 6.
   - Model split: Sonnet runs, Opus reviews the bars.
   - Output: a results section in [cluster_config_baseline.md](cluster_config_baseline.md) and an
     update of this note's verdicts.
   - Gates: G1 is X2 on the held-out half. G2 and G3 move to WC, because the judge does not exist
     yet.
3. **W8 (judge and E0) is unchanged and stays next.** X8 needs it.
4. **WB runs after W0c,** with two changes:
   - WB scores every representation variant with the W0c method (within-detector-class average
     linkage at its calibrated cut) on the 3,000-item set, not with global average linkage and not
     on the COCO sample.
   - WB drops the representation rows W0c's X3 closes.
5. **New implementation wave WC: class-partitioned residual clustering.**
   - Entry conditions: W0c's X2 passes, X7 has fixed the method form for the largest key, and X8 has
     passed on set P.
   - Model split: Opus reviews the run-record and cluster-state design (novel-candidate state, cut
     source) before code; Sonnet implements.
   - Scope:
     - a within-key partition method beside IVF in `methods/`;
     - the partition key and the `none` key;
     - cut lookup with fail-closed refusal;
     - ingest-time assignment restricted to the item's key, with abstention;
     - the novel-candidate state and cross-class suggestions, only if X4 passes;
     - deletion of the superseded IVF-as-clusters path in the same change (no shims).
   - Tests: red first. A permuted fetch gives the identical partition. An item never joins another
     key's cluster. A missing cut refuses. A `none` key item is clustered, not dropped. No class or
     box field is written (lock rule untouched). The product pass on the 3,000-item set reproduces
     W0c's held-out row within 0.01.
   - Gates: G1 is that reproduction. G2 is the X8 panels. G3 is the same 20 clusters before and after,
     in Cropwright.
   - Timing: WC is CPU code and can run in parallel with WB's GPU work. If WB changes the
     representation, WC re-calibrates the per-model cut before its G1; the method is unchanged.
6. **Later waves.** W5 (fine-grained) and W7 (fine groups) keep their per-coarse-class structure,
   which this result supports. W2 and E6 (faces) are unaffected.

## 9. Limits

- **One public set and one held-out half.** 944 images, 1,494 held-out labelled crops, 86 classes.
  Folds and subsamples agree on the direction. Values can move by a few hundredths: fold 2 recall
  0.181 against fold 1's 0.129 at nearly the same cut.
- **Person dominance.** Person makes up 78 to 81 % of the same-label pairs. Pair metrics alone would
  overstate any partition that keeps the person bucket together. Hence the BCubed and non-person
  columns.
- **The detector's label space.** The detector is trained on COCO's label space, and about half the
  images are COCO val. Agreement on nameable labels (0.93 to 0.95) is therefore optimistic for other
  domains. A project whose detector has few classes, or none, gets little or nothing from the key
  (section 4).
- **Residual-pool realism.** All crops of this project were in the residual pool. With a
  class-assigning detector, confident items leave the pool. The open-set crops are among the
  confident ones (median 0.80), so they would leave it with a wrong class before clustering sees
  them. That is a pipeline hazard outside this decision; it belongs on the auto-label loop's open-set
  work.
- **Unlabelled crops.** 1,011 of 3,961 crops carry no label, so merges involving them are invisible.
- **Weak spots of the new checks.** The novelty AUROCs rest on very small per-class samples. The
  half-subsample ranges are spreads, not intervals. The section-2 checks are scratch scripts until X1
  lands them.
- **Set P.** No set P number exists for any of this, and set P has 0 validated items. The default cut
  is unproven there (X5, X8).

## 10. Reproduce

The committed path reproduces the within-class and global pair rows (public data; bundle and labels
as built in [cluster_config_baseline.md](cluster_config_baseline.md)):

```bash
.venv/bin/python scripts/research/cluster_diag/run.py e5 --bundle <out>/vecs \
    --labels <out>/eval/labels.jsonl --coco-gt <dir>/coco_gt.json \
    --split-manifest scripts/datasets/manifests/splits/cluster_eval_3000.json \
    --confidence-cuts 1.01 --out <out>/e5_all_confidences.json
```

The section-2 checks are scratch scripts. They import the committed `scripts/research/cluster_diag`
and `scripts/bench/cluster_eval_metrics.py`, plus `src/utils/class_names.py`. Each script and its log
are in `artifacts_local/cluster_search/w0b/decision_checks/` (gitignored; X1 replaces them with
committed arms):

| Script | What it produced |
|---|---|
| `check.py` and `check2.py` | Agreement (raw and normalized), detector-only and global rows, the confusion list |
| `check3.py` | Calibrated within-class cuts, BCubed rows, open-set purity and pair split, confidences |
| `check4.py` | Half-subsample ranges |
| `check5.py` | Strata and non-person BCubed |
| `check6.py` | Novelty AUROCs |

Run each as `.venv/bin/python <script>` from any directory. The paths to the bundle, the split
manifest and the scratch COCO ground truth are written at the top of each script.
