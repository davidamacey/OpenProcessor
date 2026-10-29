# Loop cost on a fresh vehicle-heavy set: Z1, Z2 and Z3 (wave W0e)

Status: **registration committed before Z2 and Z3 were scored.** The results sections are added in a later
commit; the registration below is not edited after it. Branch `feat/cluster-w0e-experiments`, work from
`docs/cluster-config-decision` (section 0 of [cluster_config_decision.md](cluster_config_decision.md)).
No product code changed.

## Registration (fixed before any Z2 or Z3 score was computed)

Facts known when this was written: the set was fetched, ingested, exported and labelled; the split was made.
No clustering, no gate simulation and no bar was run on it. Z1 (on the two sets earlier waves already read)
had run and is direction only.

### The set

- **Source.** Open Images v7 **validation** images, CC BY 2.0, no stored rotation, with at least one box of
  the 24 classes of the earlier pins (`scripts/datasets/cluster_eval_selection.py`). 11,467 images qualify.
- **Pin.** `scripts/datasets/manifests/vehicle_eval_8000.json`, `manifest_sha256`
  `b875cd47273300f4407360fb11610b3998710ce7d156ae799d66c85b88e3b907`, seed 20261104, 8,000 images
  (`scripts/datasets/fetch_vehicle_eval.py`, selection in `scripts/datasets/vehicle_eval_selection.py`).
- **Groups** (each image in one, drawn in this order, balanced over the group's key with the shortfall of a
  short key given to the others): `openset` (243; the image holds one of the 12 animals the detector's label
  space lacks; the quota of 1,200 could not be met), `animal` (1,200; the largest box is a cat, dog, horse,
  sheep or cow), `vehicle` (5,060; the largest box is a car, truck, bus, motorcycle, bicycle or train; the
  quota of 5,200 could not be met), `rest` (1,497; the filler that brings the set to 8,000, mostly
  person-led scenes). The validation split holds too few qualifying images to fill the first and third
  quotas; this is why the set is "vehicle-heavy" rather than balanced.
- **Disjointness**, checked by the fetcher (`disjointness.json`): 0 file names and 0 SHA-256 hashes shared
  with the 3,000-item pin and its variants, with the Y1 pin and its variants, with the COCO-200 list and
  with the COCO-200 split manifest (2,075 names and 2,075 hashes compared; the COCO-200 images were
  downloaded and hashed because that list carries no hashes); 0 duplicate hashes inside the set.
- **Project.** A new throwaway project on the test stack `optest1`, API on port 5603, `/health` checked first.
  8,000 of 8,000 uploaded, 33,580 crops indexed and embedded, 0 failed. The vectors were exported read-only
  (`fetch_vectors.py`), labels built with `build_labels.py` from the datasets' own boxes at IoU 0.5: 14,961
  labelled crops, 18,619 crops without an annotation (never scored). Classes: car 7,961, person 3,295, dog
  1,655, cat 374, horse 321, truck 297, bicycle 222, motorcycle 196, cow 139, bus 91, train 71, and 291
  crops of the 12 open-set animals (`rare` stratum; no tiger crop was labelled).
- **Open-set item** (U-d): a labelled crop whose true class is one of alpaca, camel, fox, goat, kangaroo, lion,
  owl, penguin, pig, rabbit, squirrel, tiger. The detector's COCO label space has none of them.

### The split

By source image, stratified by group, with `scripts/bench/eval_split.py make-pin`, seed 20261105
(`scripts/datasets/manifests/splits/vehicle_eval_8000.json`, `manifest_sha256`
`8c4bf0317093bd95be994594424241fcd2952d2dd2a30e177ab210b38ee91e39`): 3,999 calibration and 4,001 held-out
images; 7,485 and 7,476 labelled crops (held-out half: car 3,979, person 1,661, `rare` 146). **Fold 1**
chooses the cut on the calibration half; the bars are read on the held-out half. Fold 2 (swap) is reported for
A1 to A7 only.

### Operating point

Section 6 of the decision note, unchanged: the largest target pair recall in {0.05, 0.1, 0.2, 0.3} whose
calibration-half BCubed precision is at least 0.90 both overall and on non-person crops; the cut is the most
pair-precise candidate on the calibration half that reaches the target there (`acceptance.operating_point`).
If no target qualifies the method fails every bar.

### Arms and the simulator

- **Arms**, partitions of all crops of the project, scored on the labelled crops of the held-out half:
  `ivf` (the production configuration, K = min(512, n // 6), refit on these vectors), `key_only` (one cluster
  per detector key, the `none` key included) and `within_key` (average linkage inside each key at the cut
  above).
- **Pool.** The labelled crops of the held-out half. A cluster is its members in the pool. Unlabelled crops
  take no part (no answer exists for them). All members are eligible; none is protected.
- **Gate.** `scripts/research/cluster_diag/gate_sim.py`, following `judge` and `_Node.run` of the loop
  (branch `feat/v070-assist-loop`, commit `ba9e27ad`): theta 0.9, alpha 0.05 tested at alpha / 2 per look,
  max dissent 1; first sample `required_sample(n, theta, alpha / 2, 0)`; with one dissent and too few answers
  a second look tops the sample up to `required_sample(n, theta, alpha / 2, dissent)`; pass needs dissent at
  most 1 and enough answers. The labeler is an oracle (its answer is the ground-truth class). A cluster that
  passes costs its sample and gives its other members the sample's top class; a cluster that fails, or is
  asked whole, costs every member. The loop's recursion into children is not simulated, for any arm.
- **Seeds.** 20 sampling seeds (0 to 19); every figure is the mean over them, with min and max stored.
- **Second row, not barred.** A labeler wrong independently with probability 0.10, naming another class at
  random.
- **Metrics.** U1 = decisions / pool items. U2 = share of propagated labels equal to the truth. U3 = share of
  items labelled by propagation. U2 on open-set items.

### The bars, read once, fold 1, held-out half

| ID | Bar | Evaluation rule beyond the note |
|---|---|---|
| U-a | within-key U2 >= 0.95 | INCONCLUSIVE if nothing was propagated |
| U-b | within-key U1 <= 0.5 x IVF's U1 | inclusive |
| U-c | within-key U1 <= 0.9 x key-only U1, at U2 >= 0.95 for both | FAIL if the method is below 0.95; INCONCLUSIVE if the key-only arm is below 0.95 (no arm to compare at equal precision) |
| U-d | within-key U2 on open-set items >= 0.90 | INCONCLUSIVE if fewer than 30 open-set items are propagated per run on average |

The verdict is FAIL if any bar fails, else INCONCLUSIVE if any is, else PASS. If a bar fails, the result names
the cheaper of IVF and the key-only arm among those at U2 >= 0.95. A1 to A7 (section 6) are computed on the
same half and the same cut and shown beside them: fold 1 is the acceptance direction; the reopen rule is
applied to fold 2. The bars are not moved after the read; a crash is fixed and the run repeated, and the fix is
disclosed.

### Z3: judge-gated merge inside a key (registered now, read after Z2)

- **Rule.** From the within-key partition at the Z2 cut, walk the key's dendrogram nodes with height in
  (cut, cut + window] from the lowest; a node is checked when both children are current units (cut clusters or
  accepted nodes) and both units have a labelled pool member; a judge "same" merges them, anything else blocks
  all ancestors. Truth of a pair: the two units' plurality classes in the pool are equal.
- **Judge.** A coin with the measured Haiku pair rates on COCO crops (the E0 run on branch
  `feat/cluster-w8-judge`): says "same" for a truly different pair with probability 0.030; for a truly same
  pair with 0.791 (0.850 derived from the decided-row accuracy 0.910, times 1 - 0.069 for the unsure rows,
  which count as "not same"). The harness itself is not run here: the rates are the input.
- **Cost.** One pair check = one decision. U1 including checks = (gate decisions + checks) / pool items.
- **Window** from {0.05, 0.10, 0.15}, chosen on the **calibration** half: the lowest U1 including checks among
  windows whose clean-labeler U2 >= 0.95 there. If none qualifies Z3 fails.
- **Bar**, read once on the held-out half: U1 including checks <= 0.9 x the no-merge within-key U1 at the same
  cut, and U2 >= 0.95. Sensitivity rows (false-"same" 0.015 and 0.06; a check costing 0.5 and 2 decisions) are
  added after the read and are not barred.

### What this cannot show

- The oracle labeler hides the VLM's own errors on these classes; the 0.10 row bounds a random one only.
- Pools are the labelled crops of one half; a real project holds every crop in each cluster, so real clusters
  are larger than the pool's. U1 here is therefore pessimistic for every arm.
- One fresh set, one split. The two halves share base images only through the split (none straddle).

---

# Results

Measured 2026-10-29. Code: Z1 at `6bdf87ef` (the branch point, run before the simulator was committed: its JSON stamps
that sha); Z2 and Z3 at `bec709b3` (the registration commit, code identical to `12357dab`). CPU for every score; the
ingest ran on the test stack `optest1` with Triton on GPU 2; GPU 0 and GPU 1 were not used. Shared 48-core host,
one-minute load average 9 to 48 across the runs (stored per JSON as `load_avg_start` and `load_avg_end`). Nothing here
is a timing. Raw outputs, gitignored: `artifacts_local/cluster_search/w0e/` (`z1.json`, `z2.json`, `z3.json`, logs,
`decision_checks/`); the exported bundle and the images are in a scratch directory outside the repository.

## Verdicts

| ID | Question | Result | Verdict |
|---|---|---|---|
| Z1 | Does the simulator behave, and do 3,000-crop sets have clusters big enough for a gate? | Code check by hand-computed cases and 9 mutations. On the two sets already read: key-alone has 9 and 7 clusters whose sample is at most half of them, within-key 4 and 1, IVF none (largest IVF cluster 19 and 12 crops). Direction: key-alone cheapest, IVF about 0.99 decisions per item | direction only; not acceptance |
| Z2 U-a | propagated-label precision >= 0.95 | 0.996 | **PASS** |
| Z2 U-b | decisions <= 0.5 x IVF's | 0.326 against a bar of 0.388 (IVF 0.776) | **PASS** |
| Z2 U-c | decisions <= 0.9 x the detector-key arm's, both at U2 >= 0.95 | 0.326 against a bar of 0.337 (key-only 0.374, U2 0.989) | **PASS on the registered mean, not robust** (see below) |
| Z2 U-d | U2 on open-set items >= 0.90 | no open-set item was propagated in any of the 20 runs | **INCONCLUSIVE** (registered: fewer than 30) |
| Z2 overall | | three bars pass, one cannot be read | **INCONCLUSIVE** |
| Z2 A1 to A7 | beside the U bars, fold 1 | A1 to A5, A7 pass; **A6 fails** (`rare` precision 0.399 below IVF's) | FAIL on A6 |
| Z3 | judge-gated merge inside a key | U1 including pair checks 0.333 against 0.326 with no merge (+2.2 %, the bar is -10 %) | **FAIL** |

## Z1: the simulator on the two sets already read (direction and code check only)

Both sets were read by earlier waves. The cut is the operating-point rule's on each set's calibration half
(3,000-item set: target 0.2, cut 0.385; Y1: target 0.1, cut 0.345); the pool is the held-out half. Clean oracle
labeler, mean over 20 sampling seeds. "Sample at most half" counts clusters whose no-dissent sample
(`required_sample`, 10 to 35 answers) is half of their members or less.

| Set, pool (labelled items) | Arm | Clusters | Largest | Can propagate | Sample at most half (items) | U1 | U2 |
|---|---|---|---|---|---|---|---|
| 3,000-item, held-out (1,494) | IVF | 374 | 19 | 21 | 0 (0) | 0.986 | 0.989 |
| | key alone | 69 | 410 | 31 | 9 (925) | 0.706 | 0.992 |
| | within-key | 281 | 172 | 30 | 4 (373) | 0.826 | 0.989 |
| Y1, held-out (1,104) | IVF | 335 | 12 | 10 | 0 (0) | 0.989 | 0.986 |
| | key alone | 64 | 251 | 19 | 7 (687) | 0.757 | 0.987 |
| | within-key | 354 | 65 | 14 | 1 (65) | 0.926 | 0.999 |

- **Do 3,000-crop sets have gate-sized clusters?** Barely. A cluster of `n` can propagate anything only above
  its sample: 10 members is the smallest (9 asked, 1 propagated); a sample of 35 to 54 is needed on a large
  cluster. IVF has none whose sample is half of them (K = 512 on a few thousand crops gives a dozen-member
  clusters); within-key has 1 to 4 on a held-out half, all in the person and car keys. So U1 on such a set is
  mostly "ask the cluster whole" and cannot separate the arms by gate efficiency. This is why Z2 is large.
- The supplementary pool of all labelled items of each set (not a bar) gives the same order.

## Z2: the use-case bars, read once on the fresh vehicle-heavy set

The set, the split, the operating-point rule and the evaluation rules are in the registration above. The operating
point chosen on the calibration half was **target 0.3, cut 0.515** (calibration BCubed precision 0.936, non-person
0.928). It is higher than the 0.345 of the two earlier public sets: a vehicle-heavy set has much purer keys, so
the rule can reach the largest target recall.

### U1 to U4 beside A1 to A7 (held-out half, fold 1, 7,476 labelled crops)

| Arm | Clusters in the pool | Largest | U1 (decisions per item) | U2 | U3 | Open-set items propagated, U2 on them |
|---|---|---|---|---|---|---|
| IVF refit | 482 | 110 | 0.776 (0.770 to 0.783) | 0.988 | 0.224 | 10, 1.00 |
| key alone | 39 | 3,439 | 0.374 (0.215 to 0.674) | 0.989 | 0.626 | 15.8, **0.00** |
| **within-key, cut 0.515** | 302 | 1,574 | **0.326 (0.302 to 0.525)** | **0.996** | 0.674 | 0, none |

Means over 20 sampling seeds, minimum to maximum in brackets. Second row (labeler wrong 10 % of the time, not
barred): U1 is 0.967, 0.984 and 0.980 and U3 under 0.04 for all three arms. One dissent in about 35 answers is
all the gate allows, so a labeler this noisy almost never lets a cluster pass whatever the partition is.

| Bar | Value | Bar | Result |
|---|---|---|---|
| U-a: U2 | 0.996 | >= 0.95 | PASS |
| U-b: U1 against IVF | 0.326 | <= 0.388 (0.5 x 0.776) | PASS |
| U-c: U1 against key alone (both U2 >= 0.95: 0.996 and 0.989) | 0.326 | <= 0.337 (0.9 x 0.374) | PASS |
| U-d: open-set U2 | not defined | >= 0.90, needs 30 propagated | INCONCLUSIVE |

| A bar (fold 1) | Value | Bar | Result |
|---|---|---|---|
| A1 BCubed precision | 0.938 | 0.90 | pass |
| A2 non-person precision | 0.927 | 0.90 | pass |
| A3 low-resolution precision (340 crops) | 1.000 | 0.85 | pass |
| A4 BCubed recall (IVF 0.052) | 0.428 | 0.104 | pass |
| A5 pair recall (IVF 0.010) | 0.345 | 0.029 | pass |
| A6 worst stratum of 100 or more crops | `rare` (146): 0.358 against IVF's 0.757, a gap of -0.399 | -0.02 | **fail** |
| A7 permutation ARI | 1.000 | 1.000 | pass |

Fold 2 gives the same: A1 to A5 and A7 pass, A6 fails (`rare` -0.444). The reopen rule fires on A6 only.
The IVF recall is far below its value on the small sets (0.052 against 0.165 to 0.19) because K stays at 512 while
the set grows to 33,580 crops: the incumbent is 65 crops per cluster, and 22 % of them are labelled.

### Reading the U bars

- **U-a and U-b are robust.** U2 0.996, and within-key beats half of IVF's U1 in 18 of the 20 seeds.
- **U-c passes the registered mean but is not robust.** The three arms share the dominant cost: when a large
  cluster fails the gate, every member is asked (1,500 decisions for the 1,574-crop car cluster). That is a draw,
  so U1 per seed ranges 0.30 to 0.53 for within-key and 0.21 to 0.67 for key alone. Paired by seed, within-key is
  cheaper than key alone in 12 of 20 seeds and at or under 0.9 of it in 6 of 20. The medians are 0.313 and 0.323.
  On this set the method is **not distinguishable from one gate per detector key at the effort level**; the
  mean passes by 0.011 (diagnostic, read after the fact: `decision_checks/paired.json`).
- **Why the key alone is competitive here.** The car key (3,439 of 7,476 pool items) is 0.997 pure, the person
  key 0.986, so one gate on each is almost free. Within-key's gain comes from the keys that are not pure: it
  has 21 failed clusters per run costing 1,310 decisions, against 14 failed key clusters costing 2,641 for
  key alone (`decision_checks/breakdown.json`; passed clusters cost 1,124 and 155).
- **What the key-alone arm gets wrong, and what within-key avoids.** Key alone propagates 15.8 open-set animals
  per run, all wrongly (its dog key holds 26 of them in a 0.959-pure cluster that passes in some seeds). Within-key
  propagates none: the clusters that hold them fail the gate. IVF propagates 10, all right (small pure
  clusters). **That is a protection by the gate, not evidence of precision on open-set items:** U-d cannot be read,
  because no open-set item reached a passing cluster. 4 of the 146 pool open-set items sit in a within-key
  cluster at purity 0.9 or more.
- **The simple design that wins when a bar fails** is, by the registered rule, key alone (U1 0.374 at U2 0.989,
  against IVF's 0.776). Here no bar failed, so the rule does not select it, but the paired result above says it
  is a live alternative.

## Z3: judge-gated merge (registered; read after Z2)

Cut 0.515, judge rates false-"same" 0.030 and "same" for a truly same pair 0.791. Design on the calibration
half (7,485 labelled crops, clean oracle, mean over 20 seeds):

| Window above the cut | Checks | Accepted (false "same") | Clusters | U1 without checks | U1 including checks | U2 |
|---|---|---|---|---|---|---|
| 0 (no merge) | 0 | 0 | 304 | 0.341 | 0.341 | 0.995 |
| 0.05 | 77.9 | 50.5 (0.2) | 254 | 0.352 | 0.362 | 0.995 |
| **0.10** | 87.8 | 56.8 (0.3) | 247 | 0.308 | **0.320** | 0.995 |
| 0.15 | 87.8 | 56.8 (0.3) | 247 | 0.308 | 0.320 | 0.995 |

The windows 0.10 and 0.15 give the same result: no further node became checkable between cut + 0.10 and cut + 0.15
(dendrogram nodes exist there, but their children were not accepted units). The rule chose 0.10 (the first of the
tied windows). Held-out half, read once:

| | Clusters | Checks | Accepted (false "same") | U1 without checks | U1 including checks | U2 |
|---|---|---|---|---|---|---|
| no merge | 302 | 0 | 0 | 0.326 | 0.326 | 0.996 |
| merge, window 0.10 | 246 | 81.7 | 56.1 (0.1) | 0.322 | **0.333** | 0.995 |

The bar was U1 including checks <= 0.293 (0.9 x 0.326) at U2 >= 0.95. The change is +2.2 %. **FAIL.** The merges are
sound (0.1 wrong accepts per run) and U2 holds, but they save only 1.2 % of the decisions the gate spends and the
pair checks cost 1.1 %. The cost sits in the few large clusters that fail the gate, which a merge of small
neighbours cannot help. Sensitivity (read after the result, not barred): U1 including checks is 0.332 at a
false-"same" rate of 0.015, 0.329 at 0.06, 0.327 if a pair check costs half a decision and 0.344 at two: none
reaches the bar.

## Findings

1. **The within-key partition meets the registered loop-cost bars that can be read, on a large vehicle-heavy
   set, and the simplest alternative is not clearly worse.** U-a, U-b and U-c pass on their means; U-c is a
   coin flip by seed, and U-d is unreadable. The owner's decision therefore rests on A1 to A7 (A6 fails by
   0.40) and on U-d, which this set cannot supply: it has 146 held-out open-set items, and none reached a passing
   cluster. A set with several thousand open-set crops would be needed to read U-d.
2. **The loop's gate is what protects open-set items.** Key alone wrongly labels about 16 per run; the
   within-key partition lets none through, because their clusters fail. The cost is effort (those clusters are
   asked whole).
3. **A judge-gated second-stage merge does not pay on the loop's cost** (Z3), at the measured pair rates.
4. **Seed variance is large.** The effort of every arm is dominated by whether the gate accepts the few
   largest clusters. The registered mean over 20 seeds is the reading; the per-seed range is stored.
5. **Small sets cannot measure this.** Z1: a 3,000-crop set has no IVF cluster that a gate can use.

## Limits

- **Pool.** The pool is the labelled crops of one half, 22 % of the project's crops per half; real clusters hold
  every crop, so real clusters are larger than the pool's and every arm's U1 is pessimistic. IVF is hurt most
  (its clusters are small in the pool). The supplementary pool of all labelled crops of both halves (not a bar,
  and the cut came from the calibration half) gives U1 0.670, 0.343 and 0.290 for IVF, key alone and within-key.
- **Oracle labeler.** It hides the VLM's own errors on these classes. The 0.10 row shows how little a noisy
  labeler lets through, not a calibrated VLM.
- **One set, one split.** The open-set classes have 11 to 48 labelled crops each (no tiger); U-d is unreadable.
- **Pure car key.** The key-alone arm benefits from the dominant car key being 0.997 pure at class level, which
  is what Open Images box labels give; a key mixing look-alike classes would cost it more.
- **Unlabelled crops.** 18,619 of 33,580 crops have no annotation and take no part. In a real run they are
  members of the clusters and would be asked about.
- **Z3 rates** come from 144 COCO pair rows (Haiku); on vehicle labels the judge's "different" is unreliable
  (earlier E0 note), which would make real merges rarer, not more frequent.
- **Set P.** No set P figure appears here.

## Reproduce

From the repository root, with `.venv/bin/python` (never sourced) and a CPU-only research environment; `<out>` is
an exported bundle. `OPENSEARCH_URL` points at the test stack's OpenSearch.

```bash
M=scripts/datasets/manifests
.venv/bin/python scripts/datasets/fetch_vehicle_eval.py --out <dir> --cache-dir <cache> \
    --manifest $M/vehicle_eval_8000.json --exclude-manifest $M/cluster_eval_3000.json \
    --exclude-manifest $M/cluster_eval_fresh.json --exclude-manifest $M/coco_va_200.json \
    --exclude-manifest $M/splits/coco_sample_200.json
.venv/bin/python scripts/bench/eval_split.py make-pin --pin $M/vehicle_eval_8000.json \
    --dataset vehicle_eval_8000 --seed 20261105 --out $M/splits/vehicle_eval_8000.json
# a new throwaway project on the test stack (own -p name), then:
.venv/bin/python scripts/curation/ingest_upload.py --root <dir>/images --project <slug> \
    --api-base http://127.0.0.1:<api>/curation --force-rehash
.venv/bin/python scripts/research/cluster_diag/fetch_vectors.py --opensearch-url $OPENSEARCH_URL \
    --project <slug> --out <out>/vecs
.venv/bin/python scripts/research/cluster_diag/build_labels.py --bundle <out>/vecs \
    --coco-gt <dir>/coco_gt.json --eval-manifest $M/vehicle_eval_8000.json --out-dir <out>/eval
Z=scripts/research/cluster_diag/z_run.py
.venv/bin/python $Z z1 --set pinned_3000 <3000>/vecs <3000>/eval/labels.jsonl <3000>/coco_gt.json \
    $M/splits/cluster_eval_3000.json --set fresh_y1 <y1>/vecs <y1>/eval/labels.jsonl <y1>/coco_gt.json \
    $M/splits/cluster_eval_fresh.json --out z1.json
.venv/bin/python $Z z2 --bundle <out>/vecs --labels <out>/eval/labels.jsonl --coco-gt <dir>/coco_gt.json \
    --split-manifest $M/splits/vehicle_eval_8000.json --out z2.json
.venv/bin/python $Z z3 --bundle <out>/vecs --labels <out>/eval/labels.jsonl --coco-gt <dir>/coco_gt.json \
    --split-manifest $M/splits/vehicle_eval_8000.json --z2 z2.json --out z3.json
.venv/bin/python -m pytest tests/research_scripts/test_gate_sim.py tests/research_scripts/test_u_bars.py \
    tests/research_scripts/test_merge_stage.py tests/test_vehicle_eval_selection.py -q --no-cov
```

The delete of the throwaway project is `DELETE /curation/projects/<slug>?confirm=<slug>`. The pin omits the
Open Images author column (the authors stay in `ATTRIBUTION.csv` next to the images, outside git).

## Tests

`tests/research_scripts/test_gate_sim.py` (the gate against the loop's published sample-size tables and
hand-computed clusters: a tiny cluster asked whole, a pure pass, the top-up, a dissent found only in the top-up, two
dissents, a wrong labeler, the open-set count), `test_u_bars.py` (every bar's boundary and its INCONCLUSIVE rule),
`test_merge_stage.py` (a four-leaf dendrogram: window, blocking, a wrong judge, unlabelled units, code uniqueness
across keys) and `tests/test_vehicle_eval_selection.py` (balanced draw, exclusion, groups, name and hash
disjointness). The gate's `required_sample` was also compared with the loop branch's own function on 2,232
argument combinations (0 differences). Nine mutations were applied in place and each was killed by the first test
run: maximum dissent 2, theta 0.8, no top-up, propagating the asked members too, no alpha split between looks,
the U-b factor 1.0, the U-d bar 0.80, a rejected node that does not block ancestors, and an exclusion set that is
ignored.
