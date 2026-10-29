# Class-level root choice for the auto-label loop: Z4 (wave W0f)

Status: **registration committed before Z4 was scored.** The results sections are added in a later
commit; the registration below is not edited after it. Branch `feat/cluster-z4-experiment`, work from
`docs/cluster-config-decision`, which already contains the W0e branch (`feat/cluster-w0e-experiments`;
the `--no-ff` merge was a no-op because section 0A of the decision note was written on a merge of it).
The bars come from section 0A.2 of [cluster_config_decision.md](cluster_config_decision.md). No product code
changed. Set P is not used anywhere in this note.

## Registration (fixed before any Z4 score was computed)

Facts known when this was written: the set was fetched, ingested, exported and labelled; the split was
made; the 758 open-set crops were counted. No partition, no gate or loop simulation and no bar was run on
this set. The simulator was run once on the Z2 vehicle set (read by W0e) as a code smoke check; that
output is not a result and is not reported.

### The set

- **Source.** Open Images v7 **test** split, CC BY 2.0, no stored rotation, boxes that are neither
  group-of nor depiction and cover at least 0.5 % of the frame, of the 24 classes of the earlier pins
  (`scripts/datasets/cluster_eval_selection.py`). 34,724 images qualify.
- **Pin.** `scripts/datasets/manifests/openset_test_z4.json`, `manifest_sha256`
  `8dc4f7da2e376be0b98e05088c936943af70346d8113c6ae6c486f25f090f5e5`, seed 20261106, 8,000 images
  (`scripts/datasets/fetch_vehicle_eval.py --split test`, selection in
  `scripts/datasets/vehicle_eval_selection.py` with its own quotas).
- **Groups** (each image in one, drawn in this order): `openset` (644; the image holds one of the 12
  animals the detector's label space lacks; the quota of 1,200 could not be met), `animal` (1,500;
  largest box a cat, dog, horse, sheep or cow, balanced over those five), `vehicle` (2,500; largest box
  a car, truck, bus, motorcycle, bicycle or train, balanced), `rest` (3,356; the filler to 8,000,
  balanced over the largest-box class, mostly car and person scenes).
- **The registered size could not be met.** Section 0A.2 asks for at least 2,000 open-set crops. The whole
  test split holds 1,863 boxes of the 12 classes (1,622 after the licence, rotation, depiction and
  group-of filters, 1,569 after the area filter) in 1,140 images; the earlier pins already hold 496 of
  those images, so 644 are left to draw. After ingest and labelling the set has **758 labelled open-set
  crops** (alpaca 40, camel 4, fox 56, goat 158, kangaroo 23, lion 76, owl 40, penguin 26, pig 118, rabbit
  145, squirrel 72; no tiger), 395 of them in the held-out half. The requirement is therefore not met by
  a factor of 2.6, and no Open Images test-split set can meet it. The consequence is registered below
  (B-c is INCONCLUSIVE when fewer than 30 open-set items are propagated per run); the bars are not moved.
  The Open Images train split would hold more and is not used: the registration names the test split.
- **Disjointness**, checked by the fetcher (`disjointness.json`): 0 file names and 0 SHA-256 hashes shared
  with the 3,000-item pin and its variants, the Y1 pin and its variants, the 8,000-image pin of W0e, the
  COCO-200 list and the COCO-200 split manifest (10,075 names and 10,075 hashes compared; the COCO-200
  images were downloaded and hashed because that list carries no hashes); 0 duplicate hashes inside the
  set. The 8,000-image set is a validation-split set, so its file names cannot coincide with test-split
  names in any case; the content hash check is the independent one.
- **Project.** A new throwaway project `z4openset` on the test stack `optest1` (API port 5603, `/health`
  returned `ready` before the ingest; Triton on GPU 2, the only GPU used). 7,999 of 8,000 uploaded; **one
  image (`oi_a54f93c489fad430.jpg`, vehicle group) was rejected by the product's decoder** ("Corrupt JPEG
  data: premature end of data segment") and has no crops; it was not retried and not replaced (the pin
  keeps it). 29,956 crops indexed and embedded. The vectors were exported read-only (`fetch_vectors.py`),
  labels built with `build_labels.py` from the datasets' own boxes at IoU 0.5: 14,475 labelled crops,
  15,481 without an annotation (never scored).
- **Open-set item**: a labelled crop whose true class is one of alpaca, camel, fox, goat, kangaroo, lion,
  owl, penguin, pig, rabbit, squirrel, tiger.

### The split

By source image with `scripts/bench/eval_split.py make-pin`, seed 20261107
(`scripts/datasets/manifests/splits/openset_test_z4.json`, `manifest_sha256`
`7c180e90fb57682f296ded59b726df6e89e53d55d7dd6e1ec159a3310d7493cf`): 4,000 calibration and 4,000
held-out images; 7,266 and 7,209 labelled crops, 363 and 395 open-set crops. **Fold 1** chooses the
within-key cut on the calibration half; every bar is read on the held-out half. Fold 2 is not run.

### Arms

Each arm is a partition of the labelled crops of the held-out half (the **pool**) that serves as the root
nodes of the loop; unlabelled crops take no part (no answer exists for them), all pool items are
eligible, none is protected.

| Arm | Roots | After a failed gate |
|---|---|---|
| `key` | one root per detector key (the `none` key included) | the loop's recursion |
| `key_guard` | the same | the same, and the hold-back guard before every propagation |
| `within_key` | average linkage inside each key at the cut the section-6 operating-point rule chooses on the calibration half (`acceptance.operating_point`; no qualifying target fails the arm on every bar) | the loop's recursion |
| `ivf_loop` | the production IVF configuration (K = min(512, n // 6)) refit on all vectors of the project | the loop's recursion |
| `ivf_gate` (comparator only) | the same IVF clusters | none: a failing cluster is asked whole (the Z2 convention) |

### The simulator (`scripts/research/cluster_diag/loop_sim.py`)

It follows `_Node.run`, `_decide_kind`, `_split_children` and `split_node` of the loop (branch
`feat/v070-assist-loop`, commit `ba9e27ad`, defaults theta 0.9, alpha 0.05 tested at alpha / 2 per look, max
dissent 1, branching 4, maximum depth 3, minimum node 40):

- A node's first sample is its reused answers (every answer an ancestor has for one of its members) plus a
  seeded draw up to `required_sample(n, 0.9, 0.025, 0)` in total. One dissent with too few answers tops
  the sample up once to `required_sample(n, 0.9, 0.025, dissent)`. Pass: dissent at most 1 and enough
  answers. A passing node gives every unasked member the top class.
- A failing node at depth 3, or with fewer than 40 members, goes to review. Otherwise it is split into 4
  children by complete-linkage cosine AHC (seeded k-means above 8,000 members, not reached here), sorted
  by smallest member; fewer than two children is a review leaf. Each child is a node of its own, depth + 1.
- **Decision accounting.** A decision is one labeler answer or one review. An item is decided at most once:
  asked (sampled in some node), reviewed (member of a review node, or held back by the guard), or labelled
  for free. U1 = decisions / pool items. U2 = share of propagated labels equal to the truth. U3 = share of
  items labelled by propagation. A review costs one decision, so a failed leaf costs what asking it whole
  costs.
- **Guard.** Before a passing node propagates, any unasked member whose cosine distance to the centroid of
  the node's sampled members exceeds the 95th percentile (`numpy.percentile`, linear interpolation) of the
  sampled members' own distances to that centroid is held back and goes to review (one decision each); it
  is not split further. No labels are used.
- **Labeler.** An oracle: the ground-truth class. A node's top class needs no registry entry in the barred
  reading (the project is taken to name the 12 animals; otherwise B-c could never pass).
- **Seeds.** 20 sampling seeds (0 to 19); every figure is the mean over them, with min, max and the
  per-seed values stored. The draws are seeded by `(seed, node path)`; they are not the loop's
  `random.Random` draws, because the ids differ.
- **What differs from the loop**: no budget, no failed VLM answers, no ground-truth veto, no
  cluster-refresh, and the pool is the labelled crops rather than every crop.

### The bars, read once, fold 1, held-out half, clean oracle

| ID | Bar | Evaluation rule beyond the note |
|---|---|---|
| B-a | U2 >= 0.95 | INCONCLUSIVE if nothing was propagated |
| B-b | U1 <= 0.5 x IVF's U1 | inclusive; IVF's U1 is the smaller of `ivf_loop` and `ivf_gate`, so the bar is not set by the weaker comparator |
| B-c | U2 on open-set items >= 0.90 | INCONCLUSIVE if fewer than 30 open-set items are propagated per run on average |

An arm's verdict is FAIL if any bar fails, else INCONCLUSIVE if any is, else PASS. **Adoption rule**
(`z4_bars.adoption`): the cheaper (lower U1) of `key` and `key_guard` among those that PASS, ties to
`key`; else `within_key` if it PASSES; else IVF stays. An arm that did not PASS, because it failed or
could not be read, is never adopted. The experiment's verdict is PASS if some arm passes, FAIL if every
arm fails, else INCONCLUSIVE. The bars are not moved after the read; a crash is fixed and the run
repeated, and the fix is disclosed.

Beside the bars, not barred and not used by the rule: a labeler wrong independently with probability 0.10
(naming another class at random); a **registry-closed** reading, in which the registry holds the 12
classes the detector can name and a top class outside it cannot pass (this is what the loop does for a
project whose registry lacks the 12 animals; open-set items can then only be exposed as a minority of a
registry class); the held-back counts and how many of them would have been wrong; per-key purity and
open-set counts.

### What this cannot show

- The oracle labeler hides the VLM's own errors; the 0.10 row bounds a random one only.
- The pool is the labelled crops of one half (about 24 % of the project's crops), so real clusters are
  larger and every arm's U1 is pessimistic; the loop with recursion shrinks that effect.
- One fresh set and one split; 395 open-set crops in the pool across 11 classes (2 to 87 each).
- The open-set requirement of 2,000 crops was not reachable (above), so B-c may be unreadable.
- Open Images box labels are clean at class level (the car key is almost pure); a key mixing look-alike
  classes would cost the key arms more.
