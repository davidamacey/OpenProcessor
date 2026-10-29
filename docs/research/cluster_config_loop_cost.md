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
