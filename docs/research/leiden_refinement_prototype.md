# Leiden refinement prototype (phase 1 of the graph refinement epic)

Status: research result, measured 2026-10-28 on synthetic planted partitions, CPU only. Issue #342
(epic #346). The real-data comparison on the public COCO crops and every GPU arm were **not run**;
section 8 says why and what is needed. Nothing here writes to a project or changes the API.

## 1. Question

Refinement today stays inside one cluster (`POST /clusters/refine/{cluster_id}`), so two clusters
that hold the same kind of object are never joined. Can a global kNN graph plus Leiden community
detection, started from the current partition, merge those fragments and reassign items across
boundaries without raising the false-merge rate, and what does it cost? Baselines named in the issue:
the current IVF + AHC pipeline, HDBSCAN leaf extraction, FINCH, plain Leiden and consensus Leiden.

## 2. Setup

* Throwaway CPU environment outside the repo: numpy 2, scipy, scikit-learn, `faiss-cpu` 1.15.1,
  `igraph` 1.0.0, `leidenalg` 0.12.0, `hdbscan` 0.8.44, `finch-clust` 0.2.3 (all installed from
  PyPI; FINCH needs `pynndescent` above 20,000 points, added later). Threads capped at 6, every run
  at `nice -n 19`, address space capped at 36 GB per run, one process per method and seed so the
  peak RSS belongs to that method alone.
* Scoring is the existing harness: every candidate partition is written as JSONL and scored by
  `scripts/bench/cluster_eval.py score`, with the second seed as `--second-partition`.
* The host was shared with a running benchmark (load average 35 to 45 during the runs), so wall
  times are pessimistic and noisy; compare within a table, not across tables or machines.

### Synthetic data (`scripts/research/planted.py`)

Seeded, 128-d unless stated. Classes come in groups of five siblings (sibling centre cosine about
0.73); every third group also has a "twin" class whose centre is at cosine about 0.95 from a
sibling, which no geometry separates, so about 4.5 percent false-merge pairs are a floor for every
method. On top of the clean items: 3 percent bridge points between two class centres, 2 percent
mixed items drawn in a sibling's region but labelled with another class of the group, and 5
percent uniform noise vectors, each its own singleton class (correctly unassigned). Class sizes are
log-normal, mean 150. The oracle partition (the labels themselves) scores precision 1.0, recall
1.0 and unassigned 0.05, so the ceiling is 5 percent unassigned. Sizes: 20,000, 100,000, 500,000.

### Baseline: emulated current pipeline (`scripts/research/baselines.py`)

FAISS spherical k-means with K = min(512, n // 6) trained on at most 50k rows, every item to its
nearest centroid, then complete-linkage cosine AHC at distance 0.25 inside each bucket of 4 to 8000
members (the refresh worker's IVF stage plus the refine route applied to every bucket, which is the
most favourable reading of the product). This emulation, not the product's stored output, is the
baseline on synthetic data. With sigma 0.5 at 128-d it fragments the way the real COCO partition
does (pair recall about 0.08, section 8), which is why that sigma was chosen; at 1024-d the same
sigma does not fragment (section 4), so the baseline's behaviour is a property of the geometry.

### Method under test (`scripts/research/graph_lib.py`, `leiden_refine_proto.py`)

1. kNN on L2-normalised vectors with FAISS, k = 30: exact flat index up to 150,000 rows, IVF
   (nlist = 4 sqrt(n), nprobe 16) above. Measured recall@30 of the IVF path against exact search on
   500k points: 0.991 (`knn_recall.py`, 2,000 queries).
2. Mutual-kNN graph: keep an edge only if each endpoint lists the other; edge weight is the cosine
   similarity. Optional global floor (`--min-sim`) and per-node adaptive floor
   (`--adaptive-beta`, mean minus beta times the std of the node's neighbour similarities).
   About 14.5 mutual neighbours per node survive (mean degree 14.5 at 20k, 100k and 500k; 3 to 7
   percent of nodes end up isolated).
3. Leiden with the Constant Potts Model (`leidenalg.CPMVertexPartition`, 3 iterations), either from
   singletons (`leiden_plain`) or initialised from the current partition with free reassignment
   (`leiden_init`). Resolutions gamma 0.0005 to 0.02 were swept at 20k; 0.003 and 0.01 were run at
   all sizes.
4. CHAMELEON-style validation as a post-filter (`chameleon_merge`). The fine "atoms" are the
   intersection of the current partition and the Leiden result. Adjacent atoms are merged best
   pair first, and only when both ratios clear their floors, with statistics recomputed after each
   merge: relative interconnectivity RI (cross weight per node of the smaller side over the mean
   internal weighted degree) at least 0.3 and relative closeness RC (mean cross edge weight over
   the size-weighted mean internal edge weight) at least 0.9. Internal statistics are totals, not
   min-cut bisections, so this is the usual cheap approximation of CHAMELEON. A Leiden split that
   fails validation is therefore reverted, and a merge that fails stays split.
5. A cluster with fewer than 3 members becomes unassigned (null) for every graph method and FINCH,
   so "unassigned" is comparable; HDBSCAN noise and the baseline keep their native behaviour.
6. Consensus Leiden: 5 seeds, co-association on the graph's edges, edges below 0.5 dropped, one more
   Leiden on the result. Modularity Leiden (`RBConfigurationVertexPartition`, resolution 1.0) is
   included because cuGraph's Leiden is modularity only.

Stability is the ARI between seed 0 and seed 1 over items both runs assign. The kNN is
deterministic on these sizes, so for Leiden it measures the optimiser's seed only; for the baseline
it measures k-means seeding.

## 3. Results on synthetic planted partitions (128-d)

Pair precision and recall, false-merge and false-split rates, and unassigned fraction are
`cluster_eval.py` outputs. Wall time covers kNN, graph, clustering and (where used) the CHAMELEON
filter, not file loading. Peak RSS is the whole process, vectors included.

**20k points**

| Method | Precision | Recall | False merge | False split | Unassigned | Clusters | Stability ARI | Wall s | Peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|
| Current IVF+AHC (emulated) | 0.954 | 0.080 | 0.046 | 0.920 | 0.000 | 2517 | 0.175 | 1.9 | 150 |
| Leiden CPM, singletons, g=0.003 | 0.925 | 0.863 | 0.075 | 0.137 | 0.047 | 176 | 0.990 | 3.3 | 126 |
| Leiden CPM, init from current, g=0.003 | 0.925 | 0.864 | 0.075 | 0.136 | 0.047 | 173 | 0.998 | 3.1 | 125 |
| Leiden init + CHAMELEON filter, g=0.003 | 0.953 | 0.861 | 0.047 | 0.139 | 0.091 | 132 | 1.000 | 5.2 | 123 |
| Leiden CPM init, g=0.01 | 0.950 | 0.693 | 0.050 | 0.307 | 0.081 | 338 | 0.991 | 3.6 | 126 |
| Leiden init + CHAMELEON, g=0.01 | 0.954 | 0.752 | 0.046 | 0.248 | 0.121 | 184 | 0.993 | 4.5 | 128 |
| Consensus Leiden (5 runs), g=0.003 | 0.934 | 0.863 | 0.066 | 0.137 | 0.047 | 175 | 0.999 | 15.9 | 139 |
| Leiden modularity, res 1.0 | 0.764 | 0.911 | 0.236 | 0.089 | 0.035 | 111 | 0.997 | 3.3 | 128 |
| HDBSCAN leaf, min size 5 | 0.475 | 0.091 | 0.525 | 0.909 | 0.798 | 85 | 1.000 | 90.1 | 160 |
| FINCH level 0 | 0.834 | 0.488 | 0.166 | 0.512 | 0.029 | 908 | 1.000 | 43.8 | 4349 |
| FINCH level 1 | 0.758 | 0.964 | 0.242 | 0.036 | 0.000 | 114 | 1.000 | 8.7 | 4350 |

**100k points**

| Method | Precision | Recall | False merge | False split | Unassigned | Clusters | Stability ARI | Wall s | Peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|
| Current IVF+AHC (emulated) | 0.952 | 0.077 | 0.048 | 0.923 | 0.000 | 12292 | 0.587 | 13.2 | 207 |
| Leiden CPM, singletons, g=0.003 | 0.926 | 0.788 | 0.074 | 0.212 | 0.055 | 1023 | 0.995 | 31.7 | 336 |
| Leiden CPM, init from current, g=0.003 | 0.928 | 0.787 | 0.072 | 0.213 | 0.055 | 1018 | 0.996 | 34.6 | 336 |
| Leiden init + CHAMELEON filter, g=0.003 | 0.954 | 0.799 | 0.046 | 0.201 | 0.096 | 803 | 0.997 | 32.3 | 347 |
| Leiden CPM init, g=0.01 | 0.948 | 0.577 | 0.052 | 0.423 | 0.083 | 1832 | 0.982 | 30.7 | 336 |
| Leiden init + CHAMELEON, g=0.01 | 0.954 | 0.733 | 0.046 | 0.267 | 0.122 | 1046 | 0.986 | 37.1 | 354 |
| Consensus Leiden (5 runs), g=0.003 | 0.928 | 0.786 | 0.072 | 0.214 | 0.055 | 1059 | 0.999 | 169.0 | 401 |
| Leiden modularity, res 1.0 | 0.481 | 0.904 | 0.519 | 0.096 | 0.039 | 388 | 0.991 | 58.2 | 333 |
| HDBSCAN leaf, min size 5 | not completed (see text) |||||||||
| FINCH level 0 | 0.815 | 0.402 | 0.185 | 0.598 | 0.027 | 4819 | 0.772 | 87.1 | 1362 |
| FINCH level 1 | 0.767 | 0.954 | 0.233 | 0.046 | 0.000 | 517 | 0.887 | 57.0 | 1212 |

**500k points**

| Method | Precision | Recall | False merge | False split | Unassigned | Clusters | Stability ARI | Wall s | Peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|
| Current IVF+AHC (emulated) | 0.953 | 0.082 | 0.047 | 0.918 | 0.000 | 61576 | 0.965 | 45.3 | 533 |
| Leiden CPM, singletons, g=0.003 | 0.931 | 0.815 | 0.069 | 0.185 | 0.093 | 3595 | 0.998 | 150.0 | 1231 |
| Leiden CPM, init from current, g=0.003 | 0.932 | 0.816 | 0.068 | 0.184 | 0.092 | 3586 | 0.998 | 162.5 | 1232 |
| Leiden init + CHAMELEON filter, g=0.003 | 0.953 | 0.822 | 0.047 | 0.178 | 0.095 | 3844 | 0.998 | 153.0 | 1445 |
| Leiden CPM init, g=0.01 | 0.952 | 0.621 | 0.048 | 0.379 | 0.123 | 6311 | 0.988 | 156.9 | 1234 |
| Leiden init + CHAMELEON, g=0.01 | 0.953 | 0.736 | 0.047 | 0.264 | 0.123 | 5209 | 0.990 | 219.6 | 1540 |
| Consensus Leiden (5 runs), g=0.003 | 0.933 | 0.814 | 0.067 | 0.186 | 0.093 | 3625 | 1.000 | 773.0 | 1717 |
| Leiden modularity, res 1.0 | 0.391 | 0.917 | 0.609 | 0.083 | 0.077 | 1169 | 0.986 | 186.0 | 1227 |
| HDBSCAN leaf, min size 5 | not completed (see text) |||||||||
| FINCH level 0 | 0.808 | 0.459 | 0.192 | 0.541 | 0.025 | 22796 | 0.743 | 66.1 | 1129 |
| FINCH level 1 | 0.756 | 0.959 | 0.244 | 0.041 | 0.000 | 2573 | 0.864 | 60.6 | 1128 |

HDBSCAN did not complete at 100k: seed 0 was killed after about 34 minutes of CPU time
(over the 25-minute time box), so there is no 100k or 500k HDBSCAN row. At 20k it took 90 s.

Stage split, `leiden_init` gamma 0.003 (seconds): 20k kNN 0.6, graph 0.05, Leiden 2.4; 100k kNN
17.7, graph 0.4, Leiden 16.1; 500k kNN 33.5, graph 2.9, Leiden 123.8. CHAMELEON filter on top:
0.7 s, 3.1 s and 17.0 s (gamma 0.003), 44.4 s at 500k with gamma 0.01, where the atoms are more
numerous. At 500k the graph has 3.63 million edges and the filter works on 80,594 atoms.

### Moves across cluster boundaries (`move_audit.py`)

Because the baseline is so much finer than the Leiden result, almost every change is a merge of
whole current clusters. The audit counts only items that leave their current cluster's successor
(the candidate cluster holding most of it) and compares the majority class of the old and new
cluster with the item's own label.

| Size | Method | Items moved | Fixed | Broken | Both right | Both wrong |
|---|---|---|---|---|---|---|
| 20k | Leiden init, 0.003 | 111 (0.58 %) | 16 | 32 | 28 | 35 |
| 20k | + CHAMELEON | 18 (0.10 %) | 9 | 6 | 2 | 1 |
| 100k | Leiden init, 0.003 | 1,075 (1.14 %) | 180 | 50 | 824 | 21 |
| 100k | + CHAMELEON | 716 (0.79 %) | 139 | 18 | 545 | 14 |
| 500k | Leiden init, 0.003 | 4,702 (1.04 %) | 1,462 | 348 | 2,743 | 149 |
| 500k | + CHAMELEON | 3,419 (0.76 %) | 1,112 | 166 | 2,042 | 99 |

"Fixed" means wrong before and right after (the item's class matches the new cluster's majority,
not the old one), "broken" the reverse. At 20k there are too few moves to say anything and Leiden
alone breaks more than it fixes; at 100k and 500k Leiden fixes 3.6 (100k) and 4.2 (500k) items per one it breaks
and the CHAMELEON filter raises that to 7.7 and 6.7. Counts are of moved items
that have a label and a cluster on both sides.

### Calibration sweeps (20k, 128-d, gamma, one seed)

`leiden_init` without the filter: gamma 0.0005 precision 0.814 recall 0.911; 0.001 0.859 / 0.910;
0.002 0.888 / 0.886; 0.005 0.946 / 0.810; 0.02 0.949 / 0.532. With the filter at gamma 0.003 the
RI floor moves the result (0.2: recall 0.865, 127 clusters; 0.3: 0.861, 132; 0.5: 0.845, 153) while
the RC floor (0.85, 0.90, 0.95) changed nothing, and precision stayed 0.953 to 0.954 throughout.
Adaptive per-node floor, beta 1.0: recall 0.841 (no gain); beta 0.0: recall 0.595 and 19.5 percent
unassigned (harmful); global `--min-sim 0.5`: essentially the same as no floor (recall 0.861, 131 clusters against 132). No evidence in these data for an
adaptive threshold.

## 4. Higher dimension (1024-d, the real PE-Core width)

The synthetic generator at 1024-d with the same sigma behaves differently, and that matters for
how far section 3 generalises. Distance concentration makes complete-linkage AHC merge whole
classes, so the baseline is no longer fragmented.

| Data | Method | Precision | Recall | Unassigned | Clusters | Wall s | Peak RSS MB |
|---|---|---|---|---|---|---|---|
| 100k, sigma 0.5 | Current IVF+AHC | 0.955 | 0.887 | 0.000 | 6,656 | 18.6 | 826 |
| 100k | Leiden init, 0.003 | 0.952 | 0.783 | 0.072 | 1,175 | 118.7 | 931 |
| 100k | Leiden init + CHAMELEON, 0.003 | 0.957 | 0.783 | 0.110 | 702 | 119.8 | 930 |
| 100k | Leiden init + CHAMELEON, 0.001 | 0.956 | 0.858 | 0.092 | 626 | 169.8 | 931 |
| 500k, sigma 0.5 | Current IVF+AHC | 0.956 | 0.947 | 0.000 | 33,036 | 216.9 | 3,952 |
| 500k | Leiden init, 0.003 | 0.954 | 0.763 | 0.110 | 3,621 | 447.1 | 5,130 |
| 500k | Leiden init + CHAMELEON, 0.003 | 0.956 | 0.764 | 0.110 | 3,559 | 380.5 | 5,129 |
| 20k, sigma 0.5 | Current IVF+AHC | 0.956 | 0.393 | 0.000 | 1,455 | 7.5 | 235 |
| 20k | Leiden init + CHAMELEON, 0.001 | 0.958 | 0.885 | 0.086 | 122 | 13.8 | 243 |
| 20k, sigma 0.65 | Current IVF+AHC | 0.835 | 0.000 | 0.000 | 19,442 | 12.0 | 234 |
| 20k | Leiden init, 0.003 | 0.937 | 0.827 | 0.063 | 228 | 16.8 | 246 |
| 20k | Leiden init + CHAMELEON, 0.001 | 0.953 | 0.527 | 0.281 | 119 | 76.7 | 242 |

* Where the baseline already merges classes whole (100k and 500k at sigma 0.5), Leiden at the
  128-d gamma is worse on recall (0.78 / 0.76 against 0.887 / 0.947) at equal precision; lowering
  gamma to 0.001 recovers most of it at 100k (0.858) but still does not beat the baseline. The
  best gamma moved from about 0.003 (128-d) to 0.001 or lower (1024-d, 20k sweep: gamma 0.0003 to
  0.001 recall 0.885 to 0.887, 0.005 recall 0.792, 0.01 recall 0.717, precision 0.957 to 0.959
  throughout). CPM resolution has to be calibrated per embedding distribution.
* Where the baseline is fully fragmented (sigma 0.65: 19,442 clusters for 20,000 items, recall
  0.000), Leiden from the current partition at gamma 0.003 gets precision 0.937 and recall 0.827,
  but the CHAMELEON filter with the 128-d floors (RI 0.3, RC 0.9) refuses most merges
  (recall 0.527, 28 percent unassigned) and was slower (76.7 s against 16.8 s for 20k atoms, a
  Python heap over about 20k single-item atoms). The floors need recalibration per distribution
  too, and the filter needs a faster implementation if atoms are numerous.
* kNN cost grows with the vector width and becomes the main cost at 1024-d: 97 to 100 s at 100k
  (exact), 219 to 282 s at 500k (IVF), against 18 s and 26 to 34 s at 128-d. Leiden itself was 18 s
  (100k) and 155 to 159 s (500k). Peak RSS at 500k and 1024-d was 5.1 GB (2 GB of it the vectors).

## 5. Findings

1. **A graph pass can merge the fragments the pipeline leaves.** On the fragmented baseline
   (pair recall 0.077 to 0.082 at every size), Leiden CPM gamma 0.003 lifts recall to 0.79 to 0.86
   and cuts the false-split rate from 0.92 to 0.14 to 0.21. Plain Leiden pays for it in precision
   (0.925 to 0.931 against 0.952 to 0.954, false merge 0.069 to 0.075 against 0.046 to 0.048).
2. **The CHAMELEON-style filter earns its place.** It restores precision to the baseline's
   (0.953 to 0.954, the twin-class floor) at the same recall (0.861 at 20k, 0.799 at 100k, 0.822
   at 500k against 0.864, 0.787, 0.816 unfiltered). The cost is unassigned items: 9.1 to 9.6
   percent against 4.7 to 9.3 percent unfiltered, where 5 percent is true noise. Those are items
   whose merge was refused or whose cluster is under 3 members, which is the "send to review" outcome
   the epic wants, but it is also lost coverage.
3. **Starting from the current partition does not change quality.** `leiden_plain` and
   `leiden_init` agree to within 0.003 on precision and recall at every size and take the same
   time. Initialisation is useful only if cluster-id continuity is wanted (not measured here).
4. **Moving single items across boundaries is rare and only mildly helpful.** 0.6 to 1.1 percent of
   items change cluster outside whole-cluster merges (table above). At 100k and 500k the moves are
   net positive (fixed to broken 3.6 : 1 (100k) and 4.2 : 1 (500k) unfiltered, 7.7 : 1 and 6.7 : 1 filtered); at 20k they are
   net negative. The headline gain comes from merging fragments, not from item-level reassignment.
5. **Modularity is the wrong objective.** Modularity Leiden at resolution 1.0 over-merges
   (false-merge 0.236 at 20k, 0.519 at 100k, 0.609 at 500k; the resolution limit worsens with n).
   cuGraph's Leiden is modularity only (no CPM, no initial partition), so it is not a drop-in GPU
   path for this method; that is inferred from the CPU result and the cuGraph signature, not
   measured on the GPU.
6. **Consensus Leiden adds cost, not quality.** Precision 0.934 / 0.928 / 0.933 and recall
   0.863 / 0.786 / 0.814 at 20k / 100k / 500k, equal to single Leiden (0.925 / 0.928 / 0.932), at
   5 times the time (15.9 s, 169 s, 773 s). Single-run stability is already 0.99 or better.
7. **HDBSCAN leaf and FINCH are not substitutes.** HDBSCAN labels 80 percent of 20k points noise,
   has precision 0.475 and took 90 s; it did not finish 100k in the time box (matches the earlier
   finding that these embeddings have no density gaps). FINCH has no granularity knob: level 0 is
   too fine (recall 0.40 to 0.49, false merge 0.17 to 0.19) and level 1 too coarse (recall 0.95
   to 0.96, false merge 0.23 to 0.24); it needed 4.3 GB at 20k with exact distances and its
   stability is 0.74 to 0.89 at 100k and 500k with approximate neighbours.
8. **Cost on the CPU is modest at 128-d and kNN dominated at 1024-d.** 500k points: 150 to 220 s
   and 1.2 to 1.5 GB for the Leiden and filter variants at 128-d (3.6 million graph edges), time
   growing about 4.7 times for 5 times the points (32 s at 100k). At 1024-d the same step is
   380 to 450 s and 5.1 GB. Both are a batch job size for the refresh worker, not an interactive one.
   These are loaded-host times. 1M was not run.
9. **Calibration is the open engineering problem.** gamma (best about 0.003 at 128-d, 0.001 or less
   at 1024-d) and the RI and RC floors both depend on the embedding distribution, and in one regime
   (baseline already merging classes whole) the method is not better than what exists.
10. **Adaptive thresholds and consensus did not help**; the plain mutual-kNN graph with cosine
    weights is the construction to carry forward.

## 6. Recommendation for phase 2 (#343)

**Conditional go.** The mechanism works where the current partition is fragmented, and the
evidence is clear on the synthetic regime that matches the real baseline's fragmentation. Two gates
are not yet closed and should be closed before committing phase-2 design (section 8): the
real-data run on the 539 labelled COCO crops, and a decision on the GPU arm.

If it proceeds, carry forward:

* Graph: one graph per embedding model and task, cosine kNN with k = 30 on the original vectors,
  mutual edges only, edge weight = cosine similarity, no adaptive floor. Exact search below about
  150k points, IVF (recall 0.991 at 500k, 128-d) above.
* Objective: Leiden CPM, never modularity. Resolution is calibrated per project by a log-grid sweep
  on the labelled set (candidate range 0.0005 to 0.01; 0.003 at 128-d and 0.001 or below at 1024-d
  gave the best recall at baseline-level precision), choosing the largest gamma whose precision is
  not below the current pipeline's.
* Validation: keep the CHAMELEON-style RI and RC check as a mandatory post-filter (it is what
  holds precision), recalibrate its floors per distribution, and make it faster before running it
  on atoms in the tens of thousands.
* Policy: items whose merge is refused and clusters under the minimum size go to a review state
  (about 4 to 5 percentage points of extra items above true noise here); item-level moves should
  be gated by the lock rule and recorded, since they are rare (about 1 percent) and net-positive
  only at larger sizes.
* Do not build: consensus Leiden, HDBSCAN or FINCH variants, an adaptive threshold, an
  initialisation-from-current requirement (no quality effect; add it only for id continuity).

## 7. Caveats

* Synthetic data with a hand-made geometry. The sibling, twin, bridge, mixed and noise fractions
  are my choices, the twin pairs set a 4.5 percent false-merge floor, and results depend on
  whether the baseline fragments (section 4). Section 3 is calibrated to a baseline that fragments
  like the real one.
* The baseline is an emulation that refines every bucket, not the product's stored output.
* Stability is across seeds with a deterministic kNN; it does not measure sensitivity to data
  changes. IVF kNN above 150k is approximate (recall@30 0.991 at 500k, 128-d).
* Timings were taken on a heavily loaded host. Only one seed per configuration in the sweeps.
* Only 128-d at 20k, 100k and 500k was run across all methods; 1024-d covers the Leiden and
  filter variants only.
* The CHAMELEON implementation uses total internal statistics rather than min-cut bisection.

## 8. Not done, and what is needed

* **Real COCO crops (539 labelled).** Labels and the current partition were built through the
  read-only API exactly as in `cluster_eval_baseline.md` (class set, IoU rule and strata match: 539
  labelled crops, 128 clusters, 10 classes; low-res 112, rare 113, similar classes 314). Today
  this gives pair precision 0.905, recall 0.087, false-merge 0.095, false-split 0.913, 28 mixed
  clusters, and 0.369 ARI between the two COCO projects over 531 shared crops (the numbers differ a
  little from the 2026-10-28 baseline: 0.893 and 134 clusters, because the stack's partition has
  since changed). The PE vectors for those crops are only in the project's OpenSearch index; the
  curation API does not return vectors. A read-only export was blocked by the permission system in
  this session, so no Leiden or baseline method was run on real vectors. To finish: export the
  `cocotest` item vectors with `scripts/research/fetch_project_vectors.py` (read-only search),
  then run the commands in section 9 with `--labels` and `--partition` from the COCO build.
* **GPU arms (cuGraph Leiden and ECG, cuML HDBSCAN, GPU kNN) on GPU 0.** A second environment was
  built (`venv-cluster-gpu` outside the repo) and installed cleanly from plain PyPI (the NVIDIA
  index was not needed): `cugraph-cu12` 26.10.0, `cuml-cu12` 26.10.0, `cuvs-cu12` 26.10.0,
  `cupy-cuda12x` 14.2.0, `faiss-gpu-cu12` 1.14.1; `import cugraph, cuml, cupy, faiss` and device
  detection (RTX A6000 on `CUDA_VISIBLE_DEVICES=0`) succeeded. Launching the first GPU job was
  blocked by the permission system, so there are no GPU timings, GPU memory figures or GPU quality
  numbers. `scripts/research/gpu_methods.py` (`gpu_leiden`, `gpu_ecg`, `gpu_hdbscan`, GPU kNN with
  faiss-gpu or cuVS, a memory watchdog that aborts above 40,000 MB) is written but **untested**.
* 1M points and HDBSCAN above 20k were not run.

## 9. Reproduce

Environment (outside the repo; `SCRATCH` is any directory outside the checkout):

```bash
python3 -m venv $SCRATCH/venv-cluster-research
$SCRATCH/venv-cluster-research/bin/pip install numpy scipy scikit-learn faiss-cpu igraph \
    leidenalg hdbscan finch-clust pynndescent httpx
```

Synthetic data, labels and the full CPU comparison at one size (repeat for 100000 and 500000):

```bash
PY=$SCRATCH/venv-cluster-research/bin/python; D=artifacts_local/leiden_proto
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
$PY scripts/research/planted.py --out $D/synth_20k.npz --n 20000 --seed 7        # --dim 1024 for PE width
$PY - <<'EOF'
import json, numpy as np
z = np.load('artifacts_local/leiden_proto/synth_20k.npz')
with open('artifacts_local/leiden_proto/synth_20k.labels.jsonl', 'w') as f:
    for i, l in zip(z['item_id'].tolist(), z['label'].tolist()):
        f.write(json.dumps({'item_id': i, 'label': l}) + '\n')
EOF
nice -n 19 $PY scripts/research/bench_driver.py --python $PY --vectors $D/synth_20k.npz \
    --labels $D/synth_20k.labels.jsonl --work-dir $D/out --tag 20k --timeout-s 1500 \
    --run leiden_plain_g003:leiden_plain:--gamma=0.003 \
    --run leiden_init_g003:leiden_init:--gamma=0.003 \
    --run leiden_init_chameleon_g003:leiden_chameleon:--gamma=0.003 \
    --run consensus_g003:consensus:--gamma=0.003 \
    --run leiden_modularity_r1:leiden_mod:--gamma=1.0 \
    --run hdbscan_leaf:hdbscan_leaf --run finch_L0:finch0 --run finch_L1:finch1
$PY scripts/research/tabulate_results.py $D/out/20k_results.json
$PY scripts/research/move_audit.py --labels $D/synth_20k.labels.jsonl \
    --current $D/out/20k_baseline_s0.jsonl --candidate $D/out/20k_leiden_init_chameleon_g003_s0.jsonl
$PY scripts/research/knn_recall.py --vectors $D/synth_20k.npz
```

COCO evaluation set (as in `cluster_eval_baseline.md`; the ground truth is the COCO annotations of
the sample images restricted to bicycle, bus, car, cat, cow, dog, horse, motorcycle, sheep, truck):

```bash
.venv/bin/python scripts/bench/cluster_eval_set.py build --api http://127.0.0.1:5603 \
    --project cocotest --coco-gt <coco_gt.json of the sample> --out-dir $D/cocotest
$PY scripts/research/fetch_project_vectors.py --opensearch-url http://127.0.0.1:5607 \
    --index op_prj_cocotest__items --out $D/cocotest_vectors.npz      # read-only search
$PY scripts/research/leiden_refine_proto.py run --vectors $D/cocotest_vectors.npz \
    --partition $D/cocotest/partition.jsonl --method leiden_chameleon --gamma 0.003 \
    --out $D/cocotest/leiden_chameleon.jsonl
.venv/bin/python scripts/bench/cluster_eval.py score --labels $D/cocotest/labels.jsonl \
    --partition $D/cocotest/leiden_chameleon.jsonl --out $D/cocotest/score.json
```

GPU arms (second environment, one GPU only, for example `CUDA_VISIBLE_DEVICES=0`):

```bash
python3 -m venv $SCRATCH/venv-cluster-gpu
$SCRATCH/venv-cluster-gpu/bin/pip install numpy scipy scikit-learn cugraph-cu12 cuml-cu12 \
    cuvs-cu12 cupy-cuda12x faiss-gpu-cu12 httpx
CUDA_VISIBLE_DEVICES=0 nice -n 19 $SCRATCH/venv-cluster-gpu/bin/python scripts/research/bench_driver.py \
    --python $SCRATCH/venv-cluster-gpu/bin/python --vectors $D/synth_20k.npz \
    --labels $D/synth_20k.labels.jsonl --work-dir $D/gpu --tag 20k \
    --run gpu_leiden:gpu_leiden:--gamma=1.0 --run gpu_ecg:gpu_ecg:--gamma=1.0,--consensus-runs=16 \
    --run gpu_hdbscan:gpu_hdbscan
```

Unit tests for the graph construction, the CHAMELEON merge and the generator:
`.venv/bin/python -m pytest tests/research_scripts -q --no-cov`.
