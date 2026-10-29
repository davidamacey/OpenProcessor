# Cluster configuration decision after the 3,000-item baseline (revised after W0c)

Status: decision note, first written 2026-10-29 and revised the same day after the CPU experiments
of wave W0c ([cluster_config_experiments.md](cluster_config_experiments.md), branch
`feat/cluster-w0c-experiments` at `602dbdb7`, merged into this branch). No product code changed.

Inputs:

- [cluster_config_baseline.md](cluster_config_baseline.md): E1, E5 and the diagnostics on the
  3,000-item set.
- [cluster_eval_baseline.md](cluster_eval_baseline.md): the production baselines.
- [`cluster_config_baseline_3000.json`](../benchmarks/cluster_config_baseline_3000.json).
- The consolidated research:
  `git show docs/cluster-methods-review:docs/research/clustering_research_consolidated.md`.
- The program plan: `git show docs/cluster-search-plan:docs/design/cluster_search_makemodel_plan.md`.
- The raw W0b and W0c outputs (`artifacts_local/cluster_search/w0b/`, `.../w0c/`, gitignored).
- The clustering code under `src/services/curation/clustering/`.

Naming: the consolidated note numbers the kinds K1 whole image, K2 crops, K3 faces, K4
fine-grained; the plan numbers K3 region boxes and K4 faces. This note uses names, not numbers.

## 1. Decision

**The design stays: average-linkage AHC inside each detector class.** W0c supports it on public data:

- Held-out BCubed precision is 0.934 to 0.953 overall and at least 0.914 on non-person crops, on
  every one of four splits and both folds.
- BCubed recall is 1.9 to 2.2 times IVF's.
- Permutation ARI is 1.000.
- Dense linkage fits the largest set P key: 40,016 crops in 172 s and 12.7 GiB.

**Seven things change from the first version of this note:**

1. **The cut is per (embedding model, field, project, detector key)**, not per model. The public cut
   0.346 does not carry to set P: there it puts 76.9 % of the person key and 95.3 % of the motorcycle
   key into one cluster. Each key's cut, and where the cut comes from, is recorded with the run. A key
   with no valid cut is refused, not clustered with another key's or model's cut. Set P has no
   validated items, so its per-key cuts come from the judge (X8) or from a label-free rule (Y2). It
   gets no default.
2. **Cluster size is not a quality bar.** A 30,000-crop cluster inside a person key can be correct at
   class level. Size and the shape of the dendrogram select what the judge looks at (Y3). The cut is
   chosen by judged purity (X8).
3. **No different linkage for dense keys, and no automatic split by size.** The set P result is a
   shift in distance scale; set P's vectors are tighter (mean random-pair cosine 0.63 against 0.46).
   A per-key cut addresses that. No evidence favours another linkage: complete linkage is killed, and
   Ward was never measured at class level. The dendrogram already gives a recursive split. It is kept
   for one use: splitting a cluster that the loop's sampled gate rejects. Whether that works is Y4.
   It is not an automatic size rule.
4. **Dropped:** the novel-candidate state and the cross-class merge suggestions (X4 failed: AUROC
   0.84 to 0.85 on the sheep key, no flag precision reaching 0.80, top-10 suggestion precision 0.3
   to 0.5). Open-set crops stay in their detector's bucket. The loop's sampled gate with an
   independent labeler (VLM or person) is the only path that renames them.
5. **No ingest-time assignment in the first version** (X6: 0.946 to 0.948 against the 0.95 bar).
   New items stay unassigned until the next refit. A refit also moves the partition (ARI 0.65 when
   20 % of the data is removed), so stable cluster ids must come from membership overlap (#343).
6. **Representation arms are closed for crops** (X3: whitening, k-reciprocal and DBA all lose
   precision at matched recall).
7. **The acceptance bars are corrected (section 6).** They are applied once, to a fresh public set
   (Y1), because the pinned held-out half has now been read by E1, E5 and W0c and cannot certify a
   newly chosen operating point.

## 2. Evidence (corrected)

Held-out half of the pinned split, fold 1 (cut chosen on calibration). "Non-person" means labelled
crops whose label is not person. That is one definition for every cell. The first version mixed two
definitions in three cells (W0c issue 1); the corrected values are below.

| Held-out half (1,494 labelled; 1,084 non-person) | Pair P / R | BCubed P / R / F | Non-person BCubed P / R |
|---|---|---|---|
| IVF now (production configuration, refit) | 0.734 / 0.029 | 0.792 / 0.187 / 0.302 | 0.814 / 0.252 |
| Detector class only (no clustering) | 0.906 / 0.960 | 0.786 / 0.870 / 0.826 | 0.711 / 0.825 |
| Global average linkage, cut 0.307 | 0.399 / 0.073 | 0.741 / 0.285 / 0.412 | 0.763 / 0.380 |
| Within detector class, cut 0.346 (target 0.1) | 0.924 / 0.129 | 0.943 / 0.357 / 0.518 | 0.925 / 0.461 |
| Within detector class, cut 0.381 (target 0.2) | 0.919 / 0.241 | 0.920 / 0.457 / 0.611 | 0.895 / 0.561 |
| Within detector class, cut 0.425 (target 0.3) | 0.906 / 0.312 | 0.889 / 0.533 / 0.667 | 0.852 / 0.641 |

The committed `class_prior_run.py partition` produces every row (W0c X1).

Facts that still hold from the first version, now reproduced by committed code unless marked:

- **Agreement.** Detector-class agreement is 0.820 once names go through `normalize_class_name`
  (W0c X0; `run.py` now uses it). 78 % of the disagreement is 12 Open Images animals the COCO
  detector has no class for. On labels it can name, agreement is 0.93 to 0.95.
- **Person dominance.** Person crops are 28 % of the labelled crops and 81 % of the same-label pairs.
  Pair metrics alone would reward the detector-only partition, so BCubed and non-person BCubed are
  the deciding metrics.
- **Open-set crops** (scratch checks, `artifacts_local/cluster_search/w0b/decision_checks/`):
  - The hard partition splits 48 % of their same-label pairs, against 3.1 % for labels the detector
    can name.
  - Their median detector confidence is 0.801, against 0.759 for correct calls, so 1 minus the
    confidence finds them with AUROC 0.638.
  - The crop-level novelty AUROCs reproduce in W0c X4, but the bars on them fail (section 1, item 4).

New since the first version:

- **The split spread (X9).** Over four splits and both folds, BCubed precision stays at 0.934 to 0.953
  and non-person precision at 0.914 to 0.946. BCubed recall is 0.304 to 0.382 at the target-0.1 cut.
- **Set P sizes (X7; numbers only).** The project has 91,184 crops in 37 keys: person 40,016,
  motorcycle 32,054, car 14,009, truck 2,769. Share of the key in its largest cluster at each cut:

  | Key | 0.346 | 0.30 | 0.25 | 0.20 | 0.15 |
  |---|---|---|---|---|---|
  | Person | 76.9 % | 47.5 % | 21.1 % | 7.7 % | 2.0 % |
  | Motorcycle | 95.3 % | 49.9 % | 41.6 % | 16.8 % | 6.5 % |

  These are sizes, not purity.
- **Calibration sample (X5).** A random draw of 400 labelled items in the project's own class mix
  reproduced the calibrated cut in 10 of 10 draws. Per-key capping changes the calibration target
  (it removes the person share of the pairs) and cannot work.

## 3. The production design

For a later wave; section 8 says which parts can start now.

1. **Partition key.** The item's stored `detector_class_name`, which ingest already writes through
   `normalize_class_name`. Never `detector_class_id`. Items with no detector class get the explicit
   key `none`. They are clustered within it, never dropped and never mixed into a class key.
2. **Method.** Dense average-linkage AHC per key, on input sorted by document id.
   - This replaces the residual `ahc` method: complete linkage on scikit-learn's connectivity graph,
     which is both a killed linkage and the over-merging graph form X7 measured.
   - It also replaces the residual `hdbscan` method (killed). Both are deleted in the same change.
   - IVF stays only as an index.
3. **Cut store.** One record per (embedding model, embedding field, project, key) holding the cut
   and its source:
   - `validated_sample`: chosen on at least about 400 validated items in the project's own mix (X5).
     The exact minimum is a WC parameter, from X5's 400.
   - `judge`: X8.
   - `rule`: Y2, only if it passes.
   - `public_default`: allowed only for a project whose vector geometry matches the public set it
     was calibrated on. Until Y2 defines that match, no project uses it.

   The run record lists every key's cut and source. A key without a record is refused with a typed
   error naming the key, model and field. Its items stay unassigned and the run continues for the
   other keys. Refusal is per key, never a silent fallback.
4. **Memory guard.** Dense linkage needs about n squared times 8 bytes: 12.7 GiB at 40,016 crops,
   measured. A key above a configured size limit is refused with a typed error. Extrapolated from X7,
   the limit is about 60,000 crops on a 32 GiB budget; this is not measured. The approximate average
   linkage needed above that is future work. scikit-learn's connectivity-constrained form is not it:
   it treats missing pairs as close (X7).
5. **Loop and lock rule.**
   - Clustering writes `cluster_id` only, through the existing guarded writer.
   - Class writes stay behind the loop's sampled gate with an independent labeler, and the lock rule.
   - "Cluster majority equals detector class" is true by construction under this key, so it is
     never evidence. Auto-promote stays off.
   - Region boxes are a separate pool (one class per profile) and are not changed by this.
6. **What is not built:** ingest-time assignment, the novelty state, cross-class suggestions, a
   sparse path, an automatic split by size, and per-key cut values for set P before X8.

## 4. The crops where the detector class disagrees with the label

This is unchanged from the first version, except that the suggestion stage is gone.

- **Hard partition for grouping; never a label, a lock or a veto.** A person can move any item. The
  plan's veto across confident classes is not supported, because confidence does not separate the
  detector's right calls from its wrong ones on open-set objects.
- **What happens to open-set objects.** Crops of a class the detector cannot name form fairly pure
  sub-clusters inside a wrong key (0.79 item purity). The loop's VLM labeler is what renames them.
  This depends on the loop's gate never accepting the detector name by default.
- **Fail-closed rules:** the `none` key; per-key refusal without a cut; per-key refusal above the
  memory limit; the run reports each key's share of the pool.
  - A single dominant key over many true classes (a class-agnostic detector) degenerates to the
    global cut that collapsed here.
  - The run reports it. A project whose largest key holds most of the pool and has no `judge` cut is
    refused, not clustered globally.

## 5. Kill rules, arm by arm (updated with W0c)

| Arm | Result | Verdict |
|---|---|---|
| Average linkage, global | 0.40 at recall 0.073; never matches IVF at higher recall | Killed as default |
| Average linkage within the detector class | Precision bars hold on four splits and both folds; recall bar fails narrowly at the target-0.1 cut (section 6) | **The design**; acceptance moves to the fresh set Y1 |
| Complete linkage | Below average linkage everywhere at class level | Killed for crops (still an arm for faces) |
| Leiden CPM | Below recall 0.1 only; ARI 0.64 to 0.69 | Killed as default; within-class Leiden not scheduled |
| ECG | 0.376 / 0.200 | Killed |
| HDBSCAN, FINCH | Rejected earlier on real data | Kill stands; delete the residual `hdbscan` method in WC |
| PCA whitening 256 | X3: -0.033 / -0.065 BCubed precision at matched recall | **Closed for crops** |
| k-reciprocal | X3: -0.014 (fold 2 unmatched, best -0.034) | **Closed for crops** |
| DBA | X3: -0.011 (fold 2 unmatched, best -0.007) | **Closed for crops** |
| CSLS, centring | At or below raw | Killed |
| 2b MiniBatchKMeans | 0.187 / 0.273 on items; ARI 0.553 | Killed as clusters; region boxes need their own box-level set |
| Connectivity-constrained average linkage (scikit-learn, mutual 30-NN) | X7: -0.064 BCubed precision against dense; ARI 0.27 | Killed (not average linkage over missing pairs) |
| Novelty state, cross-class suggestions | X4: AUROC 0.84 to 0.85 on sheep, flag precision at most 0.76 on the fit half, top-10 suggestion precision 0.3 to 0.5 | **Dropped** |
| IVF, K = n // 6 | 0.734 / 0.029 | Index only |

## 6. Acceptance bars, corrected

### What was wrong in the first version

These are W0c's nine issues and what each becomes.

1. **Non-person defined two ways.** Now one definition: labelled crops whose label is not person
   (section 2).
2. **"Pair recall >= 3 x 0.022" matched no measured value.** The bar now reads: three times the
   incumbent's pair recall measured on the same half and fold.
3. **The BCubed recall bar (>= 2 x incumbent).** It failed at the cut the note named, and the note's
   X9 text wrongly said the X2 bars held.
   - The bar is kept unchanged. It was not moved after it failed, and the pinned result stays a FAIL.
   - What changes is how the operating point is chosen: by a rule fixed before the fresh set is read
     (below).
4. **Per-key subsampling cannot calibrate this cut, and the COCO-200 set overlaps the 3,000-item
   set.** Calibration now uses random draws in the project's own mix (X5). Transfer to COCO-200 no
   longer certifies anything; Y1 replaces it.
5. **The sparse variant was under-specified.** Dense linkage is the method. Any sparse replacement
   must reproduce dense BCubed within 0.01 on the same keys.
6. **The stratum rule used exact comparison.** Now: strata with at least 100 labelled crops in the
   half may not fall more than 0.02 below the incumbent; smaller strata are reported only. This does
   not rescue the pinned result: `rare` (187 crops) is 0.0235 below in fold 2.
7. **X4 was under-specified.** Moot: the state and the stage are dropped.
8. **X6 seeding.** The arrivals now use their own hash namespace, as W0c did.
9. **The 10 % size rule and a global set P default.** Size is reported, not barred. Set P cuts are
   per key and come from judged purity (X8).

Also: "0 protected items moved" is vacuous on these labels (no item is protected). It moves to WC's
product tests, with protected items planted.

### Bars, fixed before Y1 is read

The incumbent is the production configuration (IVF, K = min(512, n // 6)), refit on the same vectors.
The stored production partition is reported beside it.

**The operating point**, chosen on the calibration half only: the largest target recall in {0.05,
0.1, 0.2, 0.3} whose calibration-half BCubed precision is at least 0.90 both overall and on
non-person crops. If no target qualifies, the method fails.

| ID | Bar (held-out half of Y1, fold 1 is the acceptance direction) | Source |
|---|---|---|
| A1 | BCubed precision >= 0.90 | plan 8.2 |
| A2 | BCubed precision >= 0.90 on non-person crops | person-dominance guard. This replaces the unsourced 0.88 and is stricter |
| A3 | BCubed precision >= 0.85 on low-resolution crops | plan 8.2 |
| A4 | BCubed recall >= 2 x the incumbent's (same half, same fold) | first version, unchanged |
| A5 | Pair recall >= 3 x the incumbent's (same half, same fold) | plan 8.2, incumbent measured |
| A6 | Strata with >= 100 labelled crops: BCubed precision not below the incumbent's by more than 0.02 | issue 6 |
| A7 | Permutation ARI = 1.000 on identical vectors | plan principle 9 |

**Reopen rule.** Fold 2 and three further seeded splits are reported. The decision is reopened if
any of them misses any bar by more than 0.03.

**Under these corrected bars the pinned 3,000-item result is:** fold 1 fails A4 (0.357 against
0.374, twice IVF's 0.187); fold 2 passes A4 (0.352 against twice IVF's 0.165, which is 0.330)
but fails A6 (`rare` 0.0235 below). It is recorded as a FAIL, not re-scored into a pass.

**Other bars kept:**

- Streaming assignment: accepted precision >= 0.95 on all labels. Unchanged. It is not needed until
  ingest-time assignment is proposed again.
- The X8 judge bars: section 7.

## 7. Next minimal experiments (wave W0d)

| ID | Experiment | Bar or use | Cost |
|---|---|---|---|
| Y3 | **Set P chaining check by size**, label-free, on the dendrograms X7 already built plus car and truck. Per key and cut (0.346, 0.30, 0.25, 0.20, 0.15): clusters, share of the largest cluster, share of crops in clusters of at least 3, and the merge profile of the largest cluster (merge heights over its last 20 merges, and the size of the two children it is made from). A long run of small additions just under the cut is the chaining signature; two large compact children is a real group | No bar. Size is not purity. Picks the clusters and cuts X8 judges | minutes, CPU |
| Y2 | **Label-free per-key cut rule.** The cut is factor alpha times the median within-key distance to the 10th nearest neighbour. Alpha is chosen on the 3,000 calibration half. Then check: (i) on the pinned held-out half (direction only) and on Y1 (acceptance), the rule's cut meets A1 to A7 as the calibrated cut does; (ii) on set P, the cut it gives per key (to Y3 and X8) | Adopted as cut source `rule` only if (i) passes on Y1 | under 1 h, CPU |
| Y1 | **Fresh public acceptance set.** A second seeded pin built like the 3,000-item one (COCO val2017 and Open Images, same strata quotas), with images disjoint from the 3,000-item pin and the COCO-200 project. Ingested on the test stack (own `-p` name, GPU 2), split by source image. Bars A1 to A7 applied once | The acceptance of the method | about 1 h GPU 2 ingest, then CPU |
| Y4 | **Split by descent** (optional, with Y1). On the calibration half, for clusters at the target-0.3 cut whose majority share is under 0.9: does descending one or two levels of the key's dendrogram give children of item-weighted purity at least 0.90? | Fixed beforehand: adopt the descent split for gate-rejected clusters only if at least 80 % of such clusters reach it within two levels | minutes |
| X8 | **Set P judge, contact sheets** (after W8's E0 passes for crop sheets: pair kappa >= 0.7, flag precision >= 0.8, bars to be agreed with the owner). For each key with at least 1,000 crops, try the candidate cuts from largest to smallest: 0.346, the Y2 cut, then 0.25 and 0.20. Sheets are member-weighted: pick a crop uniformly from the key, and fill its sheet with 23 seeded members of that crop's cluster. Question: "same kind of object as cell 1? flag the cells that are not". Local VLM for set P. Haiku only on an owner-approved sample, at most 20 sheets per wave (O3) | A key's cut is the largest candidate whose judged member purity has a Wilson 95 % lower bound >= 0.85 on >= 35 sheets. Stop at the first pass. Below 35 sheets: "insufficient sample", and the key stays refused. Numbers only in committed text | judge budget only |

Order: Y3 and Y2 (CPU, now). Then Y1 (one GPU 2 ingest) and Y4. Then X8 once W8's E0 has passed.
None of these needs product code.

## 8. Can WC start now?

**Yes, on a branch, for the parts that are correct whatever the open cut question decides.** No
default flip and no merge to `main` until Y1 passes and, for set P, X8 has produced per-key cuts.
That gives one PR for the feature, as the owner prefers.

Sound now:

1. **Region-box order independence.** Today `scroll_box_rows` reads in scroll order, and 2b's ARI
   under a row permutation is 0.553. Sort by (crop id, box id) before the fit, with a red-first test
   like the #372 one. This is a bug fix that is valid under any method; it may merge separately if
   the owner wants.
2. **The within-key dense average-linkage method**, with the `none` key and the partition key read
   from the stored normalized `detector_class_name`. Delete the killed residual `ahc` (complete
   linkage) and `hdbscan` methods in the same change. Run `make contracts` if a method name is on the
   wire.
3. **The cut store, and refusal per key** (no cut; cut for another model or field; key above the
   memory limit), with typed errors and a run record of cuts, sources and key sizes.
4. **Tests:**
   - A permuted fetch gives the identical partition.
   - No item joins another key's cluster.
   - Refusal in each case.
   - `none` items are clustered.
   - No class or box field is written.
   - Planted protected items are untouched.
   - The product pass on the 3,000-item vectors reproduces the research partition exactly at the
     same cut.

Wait for results before:

- Making the method the default, or deleting the IVF-as-clusters path (needs Y1).
- Any set P cut value (needs X8).
- The `rule` cut source (needs Y2).
- Split by descent (needs Y4).
- Ingest-time assignment (needs a fresh X6 above 0.95).
- Stable ids by overlap (#343). This is a prerequisite for the default flip, because a refit moves
  the partition.

**Wave order.**

1. W0c is done.
2. W0d runs Y3, Y2, Y1 and Y4.
3. W8 (judge, E0).
4. X8 on set P.
5. WC: default flip and merge.

WC coding runs in parallel from now. WB runs after Y1:

- It scores representation variants with the within-key method at its calibrated cut, on Y1's
  held-out half.
- It drops the crop post-processing rows X3 closed (whitening, k-reciprocal, DBA). Its
  preprocessing, model, TTA and face rows stay.
- If WB changes the representation, every cut is recalibrated.

## 9. Faces and whole images

Unchanged. What transfers:

- The protocol: calibration-chosen cut, held-out score, BCubed beside pair metrics, a free-prior or
  trivial baseline row, and a fresh set for acceptance once a half has been read repeatedly.
- The rules: order independence, cuts stored per model, field and key, and refusal without a cut.
- The X7 lesson: a threshold calibrated in one project's geometry does not carry to another.

What must be measured separately:

- **Faces:** E6. Identity level, ArcFace, no class prior; complete linkage is still an arm; faces
  never go to a hosted judge.
- **Whole images:** E7. The only new arm this note suggests is the set of detected classes as a soft
  key.
- **Region boxes:** a box-level set.
- **Fine-grained groups:** E9. This result supports a graph per coarse class; it says nothing about
  separating sub-classes.

## 10. Limits

- **One public set.** The pinned held-out half has now been read by three waves; hence Y1. The four
  splits share images and are not independent.
- **Person dominance.** Person is 28 % of the labelled crops and 81 % of the same-label pairs.
- **The detector's label space.** The detector is trained on COCO's label space, which most of the
  set uses.
- **Set P.** The only set P figures are sizes; no quality figure exists. Set P has 0 validated items.
- **Unlabelled crops.** 1,011 crops are never scored.
- **Scale.** The memory limit above 40,016 crops is extrapolated, not measured.
- **The judge.** X8 depends on a judge whose agreement on crops is not measured yet.

## 11. Reproduce

W0c's committed commands reproduce section 2 and every W0c figure
([cluster_config_experiments.md](cluster_config_experiments.md), Reproduce). The pair rows also
come from:

```bash
.venv/bin/python scripts/research/cluster_diag/run.py e5 --bundle <out>/vecs \
    --labels <out>/eval/labels.jsonl --coco-gt <dir>/coco_gt.json \
    --split-manifest scripts/datasets/manifests/splits/cluster_eval_3000.json \
    --confidence-cuts 1.01 --out <out>/e5_all_confidences.json
```

Some figures come only from scratch scripts in `artifacts_local/cluster_search/w0b/decision_checks/`
(gitignored), run as `.venv/bin/python <script>`:

| Script | Figures |
|---|---|
| `check3.py` | The open-set pair split (48 %) and the confidence medians |
| `check6.py` | The confidence AUROC (0.638) |
| `check4.py` | The half-subsample ranges of the first version |
