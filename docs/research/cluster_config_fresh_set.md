# Fresh-set acceptance, label-free cut rule, chaining by size and split by descent

Measured 2026-10-29 on branch `feat/cluster-w0d-experiments`, code at the commit named in each raw JSON
(`git_sha`). CPU only for every score; Triton on GPU 2 for the one ingest (test stack `optest1`, GPU 0 and
GPU 1 untouched). Shared 48-core host: the one-minute load average at the start of the final runs was 46 to 90, and the load
at the start and end of every run is stored in its JSON (`load_avg_start`, `load_avg_end`). Nothing here is a
timing.

These are experiments Y3, Y2, Y1 and Y4 of [cluster_config_decision.md](cluster_config_decision.md) section 7,
with the bars of its section 6 applied exactly as written. No bar and no operating-point rule was changed after a
result was seen. X8 (the set P judge) is not run here. No product code changed.

## Verdicts

| ID | Question | Result | Verdict |
|---|---|---|---|
| Y3 | Chaining by size on set P: person, motorcycle, car, truck at cuts 0.346 to 0.15 | Label-free sizes and merge profiles below. No bar. Large clusters at 0.346 and 0.30 on person and motorcycle are built mostly from small additions (14 to 18 of the last 20 merges add under 5 % of the cluster) | no bar; picks what X8 judges |
| Y2 | Label-free per-key cut = alpha x median 10th-neighbour distance; alpha fitted on the calibration half | alpha = 1.175. On the pinned held-out half (direction only) and on Y1, BCubed recall misses A4 and pair recall misses A5; the global cut passes both on the pinned half | **FAIL**: `rule` is not adopted as a cut source |
| Y1 | Bars A1 to A7 once on a fresh public set disjoint from both earlier sets | Fold 1: A1, A2, A3, A5, A7 pass; **A4 fails** (0.288 against 0.329) and **A6 fails** (`rare` 0.054 below the incumbent) | **FAIL** (reopen rule fires: A4 and A6 miss by more than 0.03) |
| Y4 | Do one or two descent levels make impure clusters pure (>= 80 % of them)? | 10 of 14 impure clusters on the 3,000-item calibration half (71 %); 1 of 11 on Y1's (9 %, at a degenerate cut, see below) | **FAIL** |

## Y3: chaining check by size (set P, sizes only)

Dense average linkage per key on the set P export (same vectors as the earlier scale run), cut at five heights.
"Additions" counts the last 20 merges of the largest cluster whose smaller child is under 5 % of its sibling.
A long run of additions just under the cut is the chaining signature; two large compact children are a real
group. The columns repeat the earlier shares exactly (76.9 % person, 95.3 % motorcycle at 0.346).

| Key (crops) | Cut | Clusters | Largest cluster | Crops in clusters of 3 or more | Additions among the last 20 merges | Last merge: small + large child |
|---|---|---|---|---|---|---|
| person (40,016) | 0.346 | 136 | 76.9 % | 99.8 % | 15 | 9,932 + 20,827 |
| | 0.30 | 334 | 47.5 % | 99.5 % | 18 | 371 + 18,633 |
| | 0.25 | 961 | 21.1 % | 98.6 % | 11 | 15 + 8,444 |
| | 0.20 | 2,617 | 7.7 % | 95.7 % | 5 | 1,521 + 1,543 |
| | 0.15 | 6,961 | 2.0 % | 87.0 % | 3 | 132 + 650 |
| motorcycle (32,054) | 0.346 | 24 | 95.3 % | 99.9 % | 16 | 3 + 30,534 |
| | 0.30 | 55 | 49.9 % | 99.9 % | 14 | 19 + 15,972 |
| | 0.25 | 182 | 41.6 % | 99.7 % | 15 | 115 + 13,211 |
| | 0.20 | 606 | 16.8 % | 98.9 % | 12 | 654 + 4,724 |
| | 0.15 | 2,253 | 6.5 % | 95.8 % | 6 | 33 + 2,043 |
| car (14,009) | 0.346 | 83 | 40.1 % | 99.7 % | 4 | 2,025 + 3,591 |
| | 0.30 | 215 | 13.9 % | 99.3 % | 11 | 488 + 1,461 |
| | 0.25 | 576 | 9.1 % | 97.9 % | 9 | 623 + 650 |
| | 0.20 | 1,448 | 3.5 % | 93.8 % | 3 | 6 + 483 |
| | 0.15 | 3,279 | 2.3 % | 81.1 % | 1 | 46 + 276 |
| truck (2,769) | 0.346 | 41 | 64.7 % | 99.1 % | 7 | 106 + 1,685 |
| | 0.30 | 84 | 43.7 % | 98.3 % | 6 | 30 + 1,179 |
| | 0.25 | 186 | 20.0 % | 96.1 % | 5 | 141 + 412 |
| | 0.20 | 390 | 9.4 % | 90.2 % | 6 | 63 + 198 |
| | 0.15 | 767 | 3.1 % | 77.1 % | 1 | 10 + 76 |

What it shows, as sizes only (it says nothing about purity):

- **Chaining signature.** Person at 0.30 (18 of 20), motorcycle at 0.346 to 0.25 (14 to 16 of 20) and
  car at 0.30 (11 of 20) are built mostly from small additions. Person at 0.346 splits into 9,932 + 20,827
  at the top, so its large cluster has real internal structure, but 15 of its last 20 merges are still additions.
- **Cut height trades chaining for fragmentation.** The share of crops left in clusters of fewer than 3 grows
  from 0.2 % (person, 0.346) to 13 % (0.15); for car it reaches 19 % at 0.15.
- **No bar.** Size is not purity. These are the keys, cuts and clusters X8 should judge first: person 0.30 and
  0.25, motorcycle 0.25 and 0.20.

## Y2: label-free per-key cut rule

The cut of a key is alpha times the median, over the key's crops, of the cosine distance to the 10th nearest
other crop of the same key. A key with 10 crops or fewer has no 10th neighbour: it is refused, and its crops are
unassigned (3 % of the pinned set's crops; 72 of set P's 91,184).

- **Fit.** Alpha is the swept parameter on a grid of 0.5 to 3.0 (step 0.025). The operating-point rule of
  section 6 (largest target pair recall in {0.05, 0.1, 0.2, 0.3} with calibration BCubed precision >= 0.90 overall
  and on non-person crops) chose target 0.05 and **alpha = 1.175** on the 3,000-item calibration half.
- **Pinned held-out half, fold 1 (direction only).**

  | Held-out half (1,494 labelled) | BCubed P | Non-person P | BCubed R (bar A4) | Pair R (bar A5) | A6 worst gap | Verdict |
  |---|---|---|---|---|---|---|
  | `rule`, alpha 1.175 | 0.948 | 0.931 | **0.300** (0.374) | **0.063** (0.087) | -0.005 | FAIL (A4, A5) |
  | global cut by the same rule, 0.385 (target 0.2) | 0.920 | **0.894** | 0.458 (0.374) | 0.242 (0.087) | **-0.037** | FAIL (A2, A6) |

  The rule's precision is higher and its recall lower: on the calibration half it needs a larger alpha to reach the
  same recall and then loses precision faster than the global cut does (BCubed precision 0.916 at recall 0.308
  against 0.926 at 0.427 for the global cut; the full curve is in `y2.json`, `frontier_fit_on_calibration`).
- **Y1 (acceptance direction).** With alpha frozen at 1.175: BCubed precision 0.933, recall 0.278 (bar 0.329),
  pair recall 0.090 (bar 0.115), `rare` stratum 0.071 below the incumbent. **FAIL.**
- **Set P per-key cuts (to Y3 and X8).** Alpha 1.175 times the key's own scale:

  | Key | Crops | Median 10th-neighbour distance | Rule cut | Clusters | Largest cluster | In clusters of 3 or more | Additions among the last 20 merges |
  |---|---|---|---|---|---|---|---|
  | person | 40,016 | 0.110 | 0.129 | 10,170 | 1.0 % | 79.4 % | 4 |
  | motorcycle | 32,054 | 0.087 | 0.102 | 7,536 | 1.0 % | 81.8 % | 1 |
  | car | 14,009 | 0.131 | 0.153 | 3,117 | 2.3 % | 82.4 % | 1 |
  | truck | 2,769 | 0.144 | 0.169 | 593 | 6.1 % | 83.4 % | 1 |
  | backpack | 1,409 | 0.137 | 0.160 | 383 | 7.7 % | 77.0 % | 3 |

  On the public set the same scale is 0.25 for person and 0.21 to 0.38 for the other keys with 100 or more crops;
  set P's tighter vectors give cuts of 0.10 to 0.17 against the public 0.345. The rule cuts leave no chained
  cluster, but 17 to 23 % of set P's crops sit in clusters of 1 or 2 crops. These figures are sizes; whether such
  cuts are too low is for X8.

**Verdict: FAIL.** `rule` is adopted as a cut source only if it passes on Y1; it does not. The set P cuts above
stay as candidate heights for X8, not as a cut source.

## Y1: the fresh acceptance set

**Set.** Built by `fetch_cluster_eval.py` with a new seed (20261102), excluding every file of the 3,000-item pin
(base images and variants), of the 200-image COCO project and of its split. Disjointness is checked on file name
(0 shared) and on content hash (0 shared). COCO val2017 alone could not fill the quotas once the earlier pin had
taken its images (only 22 of 180 `rare` and 29 of 220 `similar` COCO images were left), so the shortfall went to
the Open Images draw of the same group (`--backfill-from-openimages`). Training-split COCO images were not used:
the detector was trained on them.

| | Y1 | 3,000-item set |
|---|---|---|
| Images (base + variants) | 847 + 200 = 1,047 (COCO 141, Open Images 706 base) | 944 |
| Crops / labelled | 3,355 / 2,334 | 3,961 / 2,950 |
| Classes / detector keys | 80 / 72 | 86 / 79 |
| Person share of labelled | 23.1 % | 28 % |
| Split (seed 20261103, by source image) | 1,230 calibration, 1,104 held-out labelled | 1,456 / 1,494 |

Ingested into a new throwaway project on `optest1` (API on 5603, `/health` checked first), 1,047 of 1,047 uploaded
and all 3,355 crops embedded; the vectors were exported read-only, and the project was deleted afterwards. Only that
project was touched.

**Run once.** The incumbent is the production IVF configuration refit on the same vectors. Cut and operating point
are chosen on the calibration half only, and the bars are read on the held-out half (fold 1).

| Held-out half, fold 1 | BCubed P | Non-person P | Low-res P | BCubed R | Pair R |
|---|---|---|---|---|---|
| IVF, refit (incumbent) | 0.821 | 0.875 | 0.813 | 0.165 | 0.038 |
| IVF, stored production partition | 0.826 | n/a | n/a | 0.153 | 0.033 |
| Within detector class, cut 0.345 (operating point: target 0.1) | 0.927 | 0.909 | 0.951 | 0.288 | 0.128 |

| Bar | Value | Bar value | Result |
|---|---|---|---|
| A1 BCubed precision | 0.927 | 0.90 | pass |
| A2 non-person precision | 0.909 | 0.90 | pass |
| A3 low-resolution precision | 0.951 | 0.85 | pass |
| A4 BCubed recall >= 2 x incumbent | 0.288 | 0.329 | **fail** |
| A5 pair recall >= 3 x incumbent | 0.128 | 0.115 | pass |
| A6 strata of 100 or more crops, precision not below the incumbent by more than 0.02 | `rare` (251 crops): -0.054 | -0.02 | **fail** |
| A7 permutation ARI | 1.000 | 1.000 | pass |

The operating-point rule chose the same height on Y1's calibration half (0.345) as on the 3,000-item set (0.346),
so the public cut transfers between the two public sets. Both failures are of the same kind as before: the
within-class partition is far more precise than IVF everywhere except in `rare`, where IVF's tiny clusters are
almost pure (0.988) and the open-set animals Open Images adds (alpaca, camel, goat and others, which the detector
files under a COCO animal) are mixed inside the wrong key (0.933).

**Verdict: FAIL.** The bars were not moved. The reopen rule ("any further split misses any bar by more than 0.03")
fires on the pinned fold 1 itself (A4 by 0.041, A6 by 0.034 beyond its tolerance).

Fold 2 and three further seeded splits (the same seeds as the earlier extra splits) are in `y1.json`. Over all
eight split-fold cells the within-class cut passes A1, A3 and A7 in every cell, A2 in six and A5 in four; A4 fails
in all eight (best 0.316 against 0.341, a miss of 0.025) and A6 fails in six (`rare`, between 0.044 and 0.057 below
the incumbent; it passes in the two cells that chose target 0.05). In two cells the calibration half selects target 0.05 and a lower cut (0.245
and 0.275), which passes precision with room (0.971 to 0.984) and fails A4 and A5 by a wide margin.

## Y4: split by descent (calibration halves only)

For the clusters at the cut chosen for target pair recall 0.3 on the calibration half, whose majority share
over at least 5 labelled members is under 0.9: does the best of one or two levels of descent give item-weighted
purity of at least 0.90? Adoption bar, fixed beforehand: at least 80 % of such clusters.

| Set (calibration half) | Cut at target 0.3 | Judged clusters | Impure clusters | Reach 0.90 within two levels | Share | Verdict |
|---|---|---|---|---|---|---|
| 3,000-item | 0.420 | 72 | 14 | 10 (5 at level 1) | 71 % | **FAIL** |
| Y1 | 0.590 | 34 | 11 | 1 | 9 % | **FAIL** |

Limit: on Y1 the pair-precision rule that picks the cut at target 0.3 lands on 0.59, the top of the cut grid, where
the person key has merged into a few huge clusters and pair precision rises again; this is not a usable cut
(target 0.2 gives the same height). Supplementary rows at the operating cut (target 0.1, Y1 cut 0.345) give 6 of 13
(46 %); the 3,000-item set gives 6 of 9 (67 %) at 0.35 and 7 of 11 (64 %) at 0.385. None reaches 80 %. At level 2
a cluster breaks into up to four parts, so reaching purity can be trivial for small clusters (a median of 4 parts,
and 16 to 35 labelled crops end in parts of fewer than 3 labelled members); the result is therefore an upper
bound on the benefit and still fails.

## Limits

- **One fresh set.** Y1 is read once for its bars, but its fold 2 and the three extra splits share its images, so
  they are not independent evidence.
- **Y1 mix.** 706 of 847 base images are Open Images (COCO val2017 could not fill the quotas). Open Images adds
  many classes the detector cannot name, which is the hardest case for a class-partitioned method and the cause of
  the `rare` failure; the person share (23 %) is lower than on the 3,000-item set (28 %).
- **Small strata.** A3 is read on 159 low-resolution crops and A6 on strata of 159 to 819 crops. Some strata in
  Y1 (`other`: 3 crops) are never judged.
- **Y2 reads the pinned half again** (calibration to fit alpha, held-out direction only). It does not count as
  acceptance. The rule's key scale is computed from all of a key's crops in the evaluated set, labels not used.
- **Y3 and the Y2 set P rows are sizes.** No purity figure exists for set P; set P has no validated items.
- **Y4.** The impure-cluster counts are small (9 to 14); read the direction.
- **Re-run.** The first pass of each experiment ran before the code was committed. Every command was re-run at the
  committed code (`git_sha` 0ff1acdd in each JSON) and reproduced the first pass exactly (apart from the stamped sha and
  load); the first-pass JSON is kept in `old_runs_dirty_tree/`.

## Reproduce

From the repository root, with the research environment; `<out>` is an exported bundle, `<set P>` the read-only set
P export (stays outside git).

```bash
M=scripts/datasets/manifests
.venv/bin/python scripts/datasets/fetch_cluster_eval.py --out <dir> --cache-dir <cache> \
    --manifest $M/cluster_eval_fresh.json --seed 20261102 --backfill-from-openimages \
    --exclude-manifest $M/cluster_eval_3000.json --exclude-manifest $M/coco_va_200.json \
    --exclude-manifest $M/splits/coco_sample_200.json
.venv/bin/python scripts/bench/eval_split.py make-pin --pin $M/cluster_eval_fresh.json \
    --dataset cluster_eval_fresh --seed 20261103 --out $M/splits/cluster_eval_fresh.json
# ingest into a new throwaway project (own -p name, GPU 2), then export read-only
.venv/bin/python scripts/curation/ingest_upload.py --root <dir>/images --project <slug> \
    --api-base http://127.0.0.1:<api>/curation --force-rehash
.venv/bin/python scripts/research/cluster_diag/fetch_vectors.py --opensearch-url $OPENSEARCH_URL \
    --project <slug> --out <out>/vecs
.venv/bin/python scripts/research/cluster_diag/build_labels.py --bundle <out>/vecs \
    --coco-gt <dir>/coco_gt.json --eval-manifest $M/cluster_eval_fresh.json --out-dir <out>/eval
Y=scripts/research/cluster_diag/y_run.py
.venv/bin/python $Y y3 --bundle <set P>/vecs --keys person motorcycle car truck \
    --dendrograms <cache> --out y3.json
.venv/bin/python $Y y2 --bundle <3000>/vecs --labels <3000>/eval/labels.jsonl --coco-gt <3000>/coco_gt.json \
    --split-manifest $M/splits/cluster_eval_3000.json --stored-partition <3000>/eval/partition.jsonl \
    --setp-bundle <set P>/vecs --dendrograms <cache> --out y2.json
.venv/bin/python $Y y1 --bundle <out>/vecs --labels <out>/eval/labels.jsonl --coco-gt <dir>/coco_gt.json \
    --split-manifest $M/splits/cluster_eval_fresh.json --stored-partition <out>/eval/partition.jsonl \
    --pin $M/cluster_eval_fresh.json --rule-fit y2.json --out y1.json
.venv/bin/python $Y y4 --bundle <out>/vecs --labels <out>/eval/labels.jsonl --coco-gt <dir>/coco_gt.json \
    --split-manifest $M/splits/cluster_eval_fresh.json --out y4_fresh.json
.venv/bin/python -m pytest tests/research_scripts tests/test_cluster_eval_selection.py -q --no-cov
```

Raw outputs, one JSON per run with `git_sha` and the host load: `artifacts_local/cluster_search/w0d/` (gitignored):
`y3/`, `y2/`, `y1/` (`y1.json`, ingest and fetch logs, the exported bundle), `y4/`.

## Tests

`tests/research_scripts/test_chaining_by_size.py`, `test_fresh_set_acceptance.py` and
`test_split_by_descent.py` hold hand-computed cases; `tests/test_cluster_eval_selection.py` covers the exclusion and
the backfill. Nine mutations were applied in a scratch copy and each was killed by the first test run: the
exclusion set ignored, the backfill disabled, a merge child read twice in the root search, the smallest target
tried first, the BCubed recall factor lowered from 2 to 1, the stratum minimum lowered from 100 to 10, the
wrong neighbour index, an extra descent level, and the non-person condition dropped from the operating-point rule.
