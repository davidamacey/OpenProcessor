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

---

# Results

Measured 2026-10-29. Code at `71850ab0` (the registration commit; the JSON stamps that sha). CPU for every score;
the ingest ran on the test stack `optest1` with Triton on GPU 2; GPU 0 and GPU 1 were not used. Shared 48-core
host, one-minute load average 13 to 28 during the run (`load_avg_start` and `load_avg_end` in the JSON).
Nothing here is a timing. Raw output, gitignored: `artifacts_local/cluster_search/z4/z4.json` and `z4.log` (also
copied to the same path in the main checkout); the exported bundle, the images, the fetch and ingest logs are in
a scratch directory outside the repository. The run took 4 minutes; it ran once and was not repeated.

## Verdicts

| Arm | B-a U2 >= 0.95 | B-b U1 <= 0.5 x IVF's (0.417) | B-c open-set U2 >= 0.90 | Verdict |
|---|---|---|---|---|
| `key` (detector key roots + recursion) | 0.982 PASS | 0.293 PASS | 0.733 on 19.9 propagated per run: **INCONCLUSIVE** (fewer than 30) | INCONCLUSIVE |
| `key_guard` (same + hold-back guard) | 0.988 PASS | 0.377 PASS | 0.976 on 9.2 propagated per run: **INCONCLUSIVE** | INCONCLUSIVE |
| `within_key` (cut 0.425, target 0.2) | 0.990 PASS | 0.526 **FAIL** | 0.897 on 8.0 propagated per run: INCONCLUSIVE | **FAIL** |

**Z4 overall: INCONCLUSIVE. The registered rule leaves IVF in place**: no arm passed, and the two key arms cannot be
read on B-c. The reason is data, not the method: the set has 395 held-out open-set crops (the registered minimum
was 2,000 in total, 758 were reachable; see the registration), so fewer than 30 are propagated per run.

IVF's comparator for B-b is the cheaper of `ivf_loop` (U1 0.834, U2 0.979) and `ivf_gate` (U1 0.840, U2 0.981).
The within-key cut came from the operating-point rule on the calibration half: target 0.2, cut 0.425 (calibration
BCubed precision 0.926, 0.908 without persons).

## The numbers (held-out half, fold 1, 7,209 labelled crops, 395 open-set; clean oracle, 20 seeds)

| Arm | Roots | U1 decisions per item | U2 | U3 | Wrong labels per run | Open-set items propagated per run (wrong) | Held back per run (would have been wrong; open-set) |
|---|---|---|---|---|---|---|---|
| `key` | 38 | **0.293** (0.246 to 0.377) | 0.982 | 0.707 | 90.6 | 19.9 (**9.05**) | 0 |
| `key_guard` | 38 | **0.377** (0.326 to 0.447) | 0.988 | 0.623 | 53.2 | 9.2 (**0.25**) | 606.7 (37.5; 10.8) |
| `within_key` | 813 | 0.526 (0.502 to 0.584) | 0.990 | 0.474 | 33.6 | 8.0 (0.95) | 0 |
| `ivf_loop` | 477 | 0.834 (0.827 to 0.843) | 0.979 | 0.166 | 25.8 | 83.8 (2.6) | 0 |
| `ivf_gate` | 477 | 0.840 (0.830 to 0.849) | 0.981 | 0.160 | | 59.6 (U2 0.964) | 0 |

Means over 20 sampling seeds, min to max in brackets. Second row, not barred (labeler wrong 10 % of the time):
U1 is 0.941, 0.950, 0.940 and 0.968 for `key`, `key_guard`, `within_key` and `ivf_loop` with U3 at most 0.06 for every arm: one dissent in about 35 answers is all the gate allows, so a noisy labeler almost never lets a cluster pass
whatever the partition is, and no arm halves IVF's effort there. Registry-closed reading, not barred (a top class
outside the 12 detector classes cannot pass): every open-set item that is propagated is wrong by construction
(`key` 8.95 per run, `key_guard` 0.25, `within_key` 0.95); U1 barely moves (0.295, 0.379, 0.527).

## What the numbers say

1. **Effort.** Both key arms clear B-b with room: detector-key roots plus the loop's own recursion cost 0.293
   decisions per item, 0.35 times IVF's 0.834. The guard costs 0.084 more per item (0.052 to 0.126 over the 20
   seeds; always more, since it asks the same sample and adds reviews), so 0.377 is still 0.45 times IVF's.
   Within-key roots fail B-b: 0.526. They are 813 small clusters, 798 of which pass per run, and the gate asks
   a handful of members of each (3,589 asked per run), so little is saved: IVF's problem at a milder scale.
2. **Precision.** B-a passes for all three arms. The key arm's worst seed is 0.964, still above the bar.
3. **Minority exposure, the question Z4 was built for.** `key` propagates a wrong class to **9.05 open-set
   items per run on average (0 to 72 by seed)**. The heavy tail points to a few draws in which a large cluster
   passes (the dog, cat, cow and horse keys hold 71, 44, 28 and 11 open-set crops; the bear, bird and sheep keys,
   where the detector names the wrong animal, hold 78, 44 and 105 but have plurality shares of 0.18, 0.38 and
   0.29, so their roots cannot pass and are split). The guard reduces this to **0.25 per run (0 to 1 by seed)**, a drop
   of 97 %, at the cost above. Of the 606.7 members it holds back per run, 37.5 would have been wrong
   (6.2 %, against 1.8 % wrong in what `key` propagates) and 10.8 are open-set: it prevents 41 % of all
   wrong propagations and 97 % of the open-set ones (9.05 to 0.25), and sends about 16 members to review
   for each wrong label it prevents (606.7 / 37.5).
4. **B-c cannot be read.** The key arm propagated at least 30 open-set items in only 2 of 20 seeds. If it could
   be read, `key` would be at 0.733 (FAIL) and `key_guard` at 0.976 (PASS), the same ordering as in W0e where
   key alone mislabelled the open-set animals; but this is not a verdict and nothing is adopted from it. The 30
   propagated per run needed about 50 % more open-set crops in the pool, which no Open Images test-split set
   offers.
5. **The pure open-set clusters are not what is at risk.** `ivf_loop` propagates 83.8 open-set items per run
   at 0.969 (small clusters that are pure), and `ivf_gate` 59.6. The risk is
   only in a large cluster of a registry class that hides a few unknown animals.

## What follows from the registered rule

- IVF stays. Nothing ships; no code changes.
- What the evidence does support, as a direction and not as an adoption: key roots plus the loop's recursion
  are much cheaper than IVF at the same precision (B-a, B-b), and the guard is what removes the minority
  exposure at a price of 29 % more effort (0.084 on 0.293). The within-key partition buys nothing over either (B-b
  fails). Reading B-c needs a set with at least about 1,200 open-set crops (about 600 per half), which the test
  split cannot give after the earlier pins took theirs. The Open Images train split can (a repeat of Z4 on it
  would be a new registration, not a re-read of this one).

## Limits

- **B-c is unreadable**: 395 open-set crops in the pool against a registered 2,000 in total; the held-out
  half had 8 to 20 propagated open-set items per run.
- **Oracle labeler.** It hides the VLM's own errors on these classes; the 0.10 row shows how little a noisy
  labeler lets through.
- **Pool.** The pool is the labelled crops of one half (24 % of the project's crops). Real clusters hold every
  crop; every U1 is pessimistic, and an unlabelled crop that the VLM would ask about adds to it.
- **One set, one split, fold 1 only.** The key arm's open-set exposure is heavy-tailed (0 to 72 per seed), so
  the mean of 20 seeds is itself a noisy figure.
- **Class-level, clean boxes.** Open Images labels give a pure car key (0.982 plurality share); a key mixing
  look-alikes would cost the key arms more. This says nothing about make and model, or faces
  ([fine_grained_benchmark_plan.md](fine_grained_benchmark_plan.md)).
- **Sampling is not the loop's own draw**: same algorithm and seeds per `(seed, path)`, not the same ids.
- **One image failed in the product's decoder** (`oi_a54f93c489fad430.jpg`, truncated JPEG data); it has no
  crops, so 7,999 of 8,000 images are scored. PIL opens it. It was left as it is.
- **Set P.** No set P figure appears here.

## Reproduce

From the repository root, with `.venv/bin/python` (never sourced); `<dir>` and `<out>` are scratch directories
outside the repository. `OPENSEARCH_URL` points at the test stack's OpenSearch (port 5607 on `optest1`).

```bash
M=scripts/datasets/manifests
.venv/bin/python scripts/datasets/fetch_vehicle_eval.py --out <dir> --cache-dir <cache> \
    --manifest $M/openset_test_z4.json --split test --seed 20261106 \
    --quota openset=1200 --quota animal=1500 --quota vehicle=2500 --target-images 8000 \
    --exclude-manifest $M/cluster_eval_3000.json --exclude-manifest $M/cluster_eval_fresh.json \
    --exclude-manifest $M/coco_va_200.json --exclude-manifest $M/splits/coco_sample_200.json \
    --exclude-manifest $M/vehicle_eval_8000.json
.venv/bin/python scripts/bench/eval_split.py make-pin --pin $M/openset_test_z4.json \
    --dataset openset_test_z4 --seed 20261107 --out $M/splits/openset_test_z4.json
# a new throwaway project on the test stack (own -p name), then:
.venv/bin/python scripts/curation/ingest_upload.py --root <dir>/images --project <slug> \
    --api-base http://127.0.0.1:<api>/curation --force-rehash
.venv/bin/python scripts/research/cluster_diag/fetch_vectors.py --opensearch-url $OPENSEARCH_URL \
    --project <slug> --out <out>/vecs
.venv/bin/python scripts/research/cluster_diag/build_labels.py --bundle <out>/vecs \
    --coco-gt <dir>/coco_gt.json --eval-manifest $M/openset_test_z4.json --out-dir <out>/eval
.venv/bin/python scripts/research/cluster_diag/z4_run.py --bundle <out>/vecs \
    --labels <out>/eval/labels.jsonl --coco-gt <dir>/coco_gt.json \
    --split-manifest $M/splits/openset_test_z4.json --out z4.json
.venv/bin/python -m pytest tests/research_scripts/test_loop_sim.py tests/research_scripts/test_z4_bars.py \
    tests/test_vehicle_eval_selection.py -q --no-cov
```

The delete of the throwaway project is `DELETE /curation/projects/<slug>?confirm=<slug>`. The pin omits the
Open Images author column (the authors stay in `ATTRIBUTION.csv` next to the images, outside git).

## Tests

`tests/research_scripts/test_loop_sim.py` (a pure node passing on its first sample, a tiny node asked whole, a mixed
node splitting with the children reusing the parent's answers, an unsplittable node and a small node going to
review, the depth limit, the top-up with one dissent, two dissents failing both when seen at once and when the
second is found in the top-up, the registry, the guard's percentile and its end-to-end accounting, the measures'
hand-computed values, the noisy labeler's memoised answers, the AHC and k-means splitters),
`test_z4_bars.py` (every bar's boundary, the comparator choice, the adoption order and tie-break, every
INCONCLUSIVE case) and `tests/test_vehicle_eval_selection.py` (the fetcher's own quotas and target, the pin).
Twelve mutations were applied in place before scoring; eleven were killed by the first test run, and the twelfth
(the second dissent allowed) survived the first version of its test and was killed by the stronger one now in
the file: maximum depth 4, guard
percentile 50, guard comparison inverted, no reuse of the parent's answers, minimum node 20, held-back members
not sent to review, maximum dissent 2, the registry ignored, the IVF factor 0.6, the IVF comparator taking the
dearer of the two, the open-set bar 0.80, and the adoption tie going to the guarded arm.
