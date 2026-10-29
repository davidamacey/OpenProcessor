# Clustering research, consolidated: methods, evidence and the recommended stack

Status: consolidated reference and methods review, 2026-10-29. Base commit `5620f6f6` (origin/main).
It gathers what earlier notes say, what was measured, what outside apps and the literature report,
and adds one new set of measurements: geometry and transform diagnostics on the real PE-Core crop
vectors (section 4). Nothing here changes product code. The overall program plan is a separate
document. This note is the evidence that plan and the paper cite.

Hardware for the new measurements: 2 x Xeon E5-2680 v3 (48 threads), CPU only, 6 threads,
`nice 19`. The host was shared: 1-minute load average 17 to 25 during the runs. Every new number is
small-sample (544 labelled crops, 11 classes). Read it as a direction, not as a final value.

## 0. How to read this note

Every claim has one evidence tag:

| Tag | Meaning |
|---|---|
| **[R]** | Measured on real data: public COCO and Open Images crops, or set P (the owner's private photos, described only by its numbers) |
| **[S]** | Measured on synthetic planted vectors. **Speed and scale evidence only, never quality evidence.** The owner's directive: real data is hard and messy, and synthetic geometry tells us nothing about it. Quality numbers from synthetic runs are quoted only to say why they do not count |
| **[L]** | A literature or vendor figure, read at the cited URL this session or in the cited internal note. Paper-reported numbers are the authors' own and were not reproduced here |
| **[U]** | Not verified: recalled, or seen only in a secondary summary. Do not quote without checking |

The four kinds of clustering this note covers. Kinds 1 to 3 exist in code
([cluster_types.md](cluster_types.md)). Kind 4 is a new requirement:

| Kind | What is grouped | Level | Space today |
|---|---|---|---|
| K1 whole image | one vector per image (albums, scenes, near-duplicates) | scene or topic | MobileCLIP2-S2, 512 |
| K2 detection crops / items | crops of any class, from many viewpoints (curation 2a items, 2b region boxes) | class | PE-Core-L14-336, 1024 |
| K3 faces | detected faces grouped into persons | identity | ArcFace `w600k_r50` (InsightFace `buffalo_l`), 512 |
| K4 fine-grained groups | e.g. vehicle make/model inside one detector class, over millions of images | sub-class (between class and identity) | none yet; PE-Core by default |

## 1. Summary

1. **The headline real-data numbers are worse than the brief's "precision about 0.95".** That figure
   is synthetic [S]. On real crops the production partition has pair precision **0.696** and recall
   **0.022** on the 3,000-item public set: 2,950 labelled crops, 86 classes, false merges 0.400 on
   low-resolution crops. On the easier 200-image COCO project it has precision 0.89 to 0.905 and
   recall 0.08 to 0.09 [R] ([cluster_eval_baseline.md](cluster_eval_baseline.md), and the #341 branch
   version of it).
2. **The fragmentation comes mostly from configuration, not from the embeddings.** IVF uses
   K = min(512, n // 6), about 6 to 8 crops per cluster whatever the number of classes. That forces
   pair recall near zero. It also makes the partition depend on row order (ARI 0.306 under a row
   permutation) [R]. A single global average-linkage AHC on the raw vectors, with no IVF, reaches
   precision 0.92 at recall 0.20 on the COCO crops [R]. That equals the tuned in-sample
   Leiden + CHAMELEON result (0.944 / 0.195, with 9.8 percent unassigned), and this simpler method
   leaves nothing unassigned.
3. **Above recall of about 0.3, the embeddings as fed to the model are the limit.** On real crops,
   only 82 percent of a crop's 10 nearest labelled neighbours share its class (P@10 = 0.82, on 11
   easy classes). The figure is 0.78 for crops under 32 px and 0.74 for crops more elongated than
   2:1 [R]. Every graph or linkage method bleeds once it has to use those cross-class edges, so
   chaining is a property of the neighbourhoods, not of the algorithm.
4. **A preprocessing defect is a likely, testable cause.** PE crops are resized on the shorter edge
   and then center-cropped (`src/utils/pe_preprocess.py`). An elongated crop loses its ends: the
   median crop keeps 69 percent of its long side, and 21 percent of the labelled crops are more
   elongated than 2:1 [R]. Whether letterboxing or a context margin fixes the elongated-crop deficit
   needs a re-encode (experiment E2). The deficit is confounded with class: buses, trains and people
   are the elongated ones.
5. **The space is strongly anisotropic and mildly hubby.** The mean cosine between random crops is
   0.50 for COCO and 0.63 for set P. The norm of the mean vector is 0.71 to 0.79, and the effective
   rank is only 104 to 122 out of 1,024. Hubness skew is 2.4 to 2.6 [R]. One fixed cosine threshold
   therefore means different things in different projects: the median 1-NN cosine is 0.87 for COCO
   and 0.92 for set P.
6. **Cheap post-hoc fixes help at cluster level, not at neighbour level.** PCA whitening to 256
   dimensions, fitted on an independent real pool (set P), leaves P@10 unchanged (0.81). It raises
   average-linkage precision at matched recall by 0.03 to 0.11. On a held-out image split it had
   higher precision in all six fold-and-target cells, but lower recall in two of them, so it is not
   a clean win yet [R]. k-reciprocal re-ranking cuts hubness skew from 2.57 to 0.69 and
   adds 0.026 P@1. Centering alone does nothing. All of these gains are small next to the
   preprocessing and embedding questions.
7. **What the earlier research recommended still holds, but with less weight than the synthetic runs
   suggested.** The research covered Leiden CPM with mutual kNN, CHAMELEON validation, no UMAP and no
   HDBSCAN. Modularity, FINCH, HDBSCAN and UMAP are rejected on real data [R]. Leiden is not better
   than plain average linkage at low recall on real data. It does keep more precision at high recall:
   0.776 at recall 0.51, against 0.40 for average linkage at 0.59 [R].
8. **Faces (K3) can borrow a production-tested design directly.** Immich runs the same ArcFace
   weights we ship (`buffalo_l`) with a DBSCAN-derived incremental algorithm: cosine distance 0.5,
   at least 3 faces for a core point, detection score at least 0.7 [L]. Our union-find at 0.7 has
   no core-point rule and no quality gate, and has never been scored. The learned-linkage literature
   (STAR-FC, Ada-NETS, Hi-LANDER) reports pairwise F of about 0.92 on MS1M splits [L]. It needs
   training on MS-Celeb-1M, which Microsoft withdrew (section 5.5).
9. **Fine-grained groups (K4) need labels.** Even trained category-discovery methods reach only 53.8
   percent clustering accuracy on Stanford Cars, with half the classes labelled (SimGCD, DINO
   ViT-B/16) [L]. The realistic K4 stack has three steps: cluster at class level, propagate from a
   few labelled or VLM-named seeds, and keep an explicit "unknown" bucket.
10. **What is missing.** No real-data number exists yet for faces, whole images, fine-grained
    groups, region boxes (2b), other embedding models (DINOv2, SigLIP 2), the preprocessing fix, any
    held-out evaluation larger than 544 crops, or the VLM judge's agreement with ground truth.
    Section 8 lists the experiments that close these gaps, all on real data with a held-out split.

## 2. What is deployed, per kind (short; details in [cluster_types.md](cluster_types.md))

| Kind | Mechanism | Granularity it produces | Ever scored on labels? |
|---|---|---|---|
| K1 | FAISS IVF over `visual_search_*`, default 1,024 cells (global) | a cell partition, not albums | no |
| K2a | FAISS spherical k-means, K = min(512, n // 6), then on-demand complete-linkage AHC at distance 0.25 | about 6 to 8 items per cluster | yes: 3,000-item set and COCO project [R] |
| K2b | `MiniBatchKMeans`, k = max(8, round(n / 800)) | about 800 boxes per cluster: **the opposite extreme of 2a** | no |
| K3a | FAISS IVF, 2,048 cells | cells, not people | no |
| K3b | union-find at cosine 0.7 (single linkage, dense n x n) | connected components | no |
| K4 | none | | |

2a and 2b are badly mis-scaled in opposite directions: about 6 items per cluster against about
800. Neither granularity was derived from data.

## 3. Measured evidence

### 3.1 Real data [R]

**Production partition (IVF, refresh worker), 3,000-item public set** (COCO val2017 plus Open Images,
944 images, 3,961 crops, 2,950 labelled, 86 classes; branch `fix/v070-341-work`,
`docs/research/cluster_eval_baseline.md`):

| | All | Low res | Mixed scene | Near-dup | Rare | Similar classes |
|---|---|---|---|---|---|---|
| Pair precision | 0.696 | 0.601 | 0.723 | 0.896 | 0.944 | 0.882 |
| Pair recall | 0.022 | 0.042 | 0.013 | 0.244 | 0.278 | 0.094 |
| Items outside their cluster's majority class | 0.208 | 0.251 | 0.169 | 0.053 | 0.036 | 0.074 |

The same vectors in a permuted row order give ARI 0.306. A re-ingest gives ARI 0.356. People make
up 28 percent of the labelled crops and dominate the low-resolution and mixed-scene strata. The
vectors of this project are no longer on the test stack, so the new diagnostics below could not
be run on it (E1 re-creates them).

**COCO 200-image project** (1,364 crops, 539 to 544 labelled, 11 classes; main
[cluster_eval_baseline.md](cluster_eval_baseline.md) and the Leiden prototype, branch
`research/v080-342-leiden-prototype`, `docs/research/leiden_refinement_prototype.md` section 3b):

| Method | Precision | Recall | Unassigned | Note |
|---|---|---|---|---|
| Current pipeline | 0.905 | 0.088 | 0 | |
| Leiden CPM init, gamma 0.1 | 0.905 | 0.234 | 0.032 | |
| Leiden + CHAMELEON (RI 0.5, RC 0.95), gamma 0.03 | 0.944 | 0.195 | 0.098 | floors chosen on the same labels (in-sample) |
| Leiden plain, gamma 0.01 | 0.745 | 0.524 | 0.020 | |
| Consensus Leiden | 0.758 | 0.526 | 0.019 | no gain over a single run |
| Modularity Leiden | 0.709 | 0.579 | 0.009 | reject |
| HDBSCAN leaf | 0.920 | 0.031 | 0.766 | reject |
| FINCH level 0 / 1 | 0.852 / 0.695 | 0.041 / 0.251 | 0.178 / 0 | reject |

**Earlier production corpus** (about 350k real crops, 2026-05 to 09,
[clustering_methods.md](clustering_methods.md)): UMAP to 50-d then AHC put 97.7 percent of the pool
in one cluster. HDBSCAN on raw vectors gave either one cluster (95 to 100 percent) or 100 percent
noise across `min_cluster_size` 5 to 200. sklearn complete-linkage AHC at 347k ran about an hour
and peaked near 124 GB.

### 3.2 Synthetic [S]: speed and scale only

| What | Number | Source |
|---|---|---|
| IVF pass, 1M vectors (fetch, fit, write-back) | 596 s wall, about 8 KB per crop overhead, linear | [v070_measurement_log.md](v070_measurement_log.md) |
| kNN (k 30) + Leiden CPM, 500k, 128-d | 150 to 220 s, 1.2 to 1.5 GB | Leiden prototype section 3 |
| Same, 500k, 1024-d | 380 to 450 s, 5.1 GB; kNN dominates (219 to 282 s IVF kNN) | Leiden prototype section 4 |
| CHAMELEON filter on 20k single-item atoms | 76.7 s (a Python heap; needs a faster implementation) | Leiden prototype section 4 |
| CPU HDBSCAN, 100k | did not finish in 25 min | Leiden prototype section 3 |
| cuML HDBSCAN, 500k | 20 s, 767 MB GPU | Leiden prototype section 3c |
| cuGraph ECG, 20k | 7.2 s, 1.8 GB GPU; nothing above 20k | Leiden prototype section 3c |

The synthetic quality results do not count as evidence: Leiden recall 0.86, CHAMELEON holding
precision at 0.953, ECG at 0.953 / 0.899. The prototype's own 1024-d runs show why. At the real
PE-Core width with the same noise, the baseline stops fragmenting, so the synthetic geometry
decides the winner. On real crops the same Leiden pass traded precision for recall instead.

## 4. Diagnosis: why fragmentation and chaining happen with these embeddings

### 4.1 New diagnostics on real vectors (this note) [R]

Data:

- 1,364 PE-Core crop vectors of the public COCO project, read through a read-only search on the
  test stack. 544 crops carry a COCO label (the current labels file, five more than the 539 in the
  baseline doc).
- 40,000 set P crop vectors (of 91,184), used only to fit the statistics of each transform: mean,
  PCA whitening and CSLS reference neighbourhoods. The transforms are therefore fitted on data
  independent of the scored set.

Box geometry for the aspect and size strata comes from the COCO annotation file.

**Geometry of the space**

| | COCO crops (1,364) | Set P crops (20,000 sample) |
|---|---|---|
| Mean cosine of random pairs (5th to 95th percentile) | 0.50 (0.30 to 0.72) | 0.63 (0.45 to 0.80) |
| Norm of the mean vector | 0.71 | 0.79 |
| Variance share of PC1 / of the top 10 PCs | 0.136 / 0.394 | 0.126 / 0.449 |
| Effective rank (entropy of the PCA spectrum), of 1,024 | 122 | 104 |
| Median cosine to the 1st / 10th neighbour | 0.87 / 0.77 | 0.92 / 0.88 |
| Hubness: skewness of the 10-occurrence count N10 | 2.57 | 2.42 |
| Anti-hubs (never in anyone's 10-NN) | 2.9 % | 3.7 % |
| Largest N10 (expected 10) | 89 | 119 |

**Neighbour purity: the ceiling for any graph or linkage method.** Neighbours are searched among
the labelled crops only.

| Similarity | P@1 | P@10 | 10-NN vote acc | P@10 low-res | P@10 elongated > 2:1 | P@10 aspect <= 2:1 | P@10 side >= 96 px | Hub skew |
|---|---|---|---|---|---|---|---|---|
| raw cosine | 0.877 | 0.821 | 0.884 | 0.781 | 0.741 | 0.843 | 0.871 | 2.57 |
| centered (set P mean) | 0.879 | 0.812 | 0.882 | 0.767 | 0.750 | 0.829 | 0.862 | 2.23 |
| PCA 256, no whitening | 0.890 | 0.809 | 0.888 | 0.778 | 0.741 | 0.827 | 0.849 | 2.43 |
| PCA-whitened 256 (set P fit) | 0.890 | 0.809 | 0.871 | 0.781 | 0.767 | 0.820 | 0.846 | 1.49 |
| PCA-whitened 128 | 0.871 | 0.784 | 0.871 | 0.762 | 0.717 | 0.801 | 0.815 | 1.70 |
| CSLS local scaling, k 10 | 0.886 | 0.810 | 0.871 | 0.775 | 0.760 | 0.823 | 0.848 | 1.63 |
| k-reciprocal re-rank (k1 20, k2 6, lambda 0.3) | 0.903 | 0.838 | 0.879 | 0.786 | 0.763 | 0.858 | 0.890 | 0.69 |
| DBA / alpha-QE (k 5, alpha 3) | 0.904 | 0.829 | 0.879 | 0.787 | 0.754 | 0.850 | 0.881 | 1.14 |

Label facts:

- 21.1 % of the labelled crops are more elongated than 2:1, and 3.5 % more than 3:1.
- 20.6 % have a short side under 32 px.
- The median center-crop keeps 0.69 of the long side.
- **The detector's own class agrees with the COCO label for 92.1 % of crops.** This is all crops; the
  production residual pool holds only the unconfident ones, where agreement will be lower.

**Clustering curves.** All 1,364 crops are clustered and the 544 labelled ones are scored. Each cell
is the best precision on a threshold sweep subject to a minimum recall. That is an oracle threshold,
so read it as a comparison of curves, not as an operating point. The Leiden column is CPM on the
mutual 30-NN graph, with resolution swept as a fraction of the mean edge weight and clusters under
3 left unassigned.

| Similarity | Avg-AHC P at R >= 0.2 / 0.3 / 0.5 | Complete-AHC P at R >= 0.2 / 0.3 / 0.5 | Leiden CPM P at R >= 0.2 / 0.3 / 0.5 |
|---|---|---|---|
| raw | 0.920 / 0.856 / 0.404 | 0.878 / 0.730 / 0.301 | 0.925 / 0.889 / 0.776 |
| centered | 0.855 / 0.524 / 0.405 | 0.888 / 0.731 / 0.296 | 0.919 / 0.868 / none |
| PCA-whitened 256 | **0.962 / 0.906 / 0.541** | 0.932 / 0.539 / 0.247 | 0.927 / **0.918 / 0.778** |
| CSLS (centered) | 0.951 / 0.609 / 0.456 | 0.804 / 0.514 / 0.263 | 0.908 / 0.891 / none |
| k-reciprocal | 0.933 / 0.836 / **0.752** | 0.877 / 0.877 / 0.326 | 0.873 / 0.778 / none |
| DBA | 0.909 / 0.823 / 0.474 | 0.902 / 0.675 / 0.392 | 0.886 / 0.811 / 0.798 |

Unassigned share on the raw Leiden curve: 1.5 to 4.4 % up to gamma 0.2, rising to 42 % at the highest
resolution, where precision reaches 0.96 at recall 0.09. Average-linkage AHC assigns everything.

**Held-out check (threshold chosen on one half of the images, scored on the other; two folds).**
Average-linkage AHC, test-fold precision / recall:

| Target recall on the dev half | Raw, fold 1 | Raw, fold 2 | Whitened 256, fold 1 | Whitened 256, fold 2 |
|---|---|---|---|---|
| >= 0.15 | 0.943 / 0.191 | 0.958 / 0.114 | 0.976 / 0.249 | 0.959 / 0.115 |
| >= 0.20 | 0.887 / 0.296 | 0.904 / 0.155 | 0.953 / 0.341 | 0.935 / 0.148 |
| >= 0.30 | 0.796 / 0.371 | 0.836 / 0.263 | 0.905 / 0.440 | 0.891 / 0.175 |

Whitening has higher precision in all six fold-and-target cells. Recall is higher in fold 1 and
lower in fold 2 at the two higher targets (0.148 against 0.155, 0.175 against 0.263), so this is
not a clean win. The spread between folds is large (recall 0.15 against 0.34 at one setting), so with about 270 labelled
crops per half this is a direction, not a value. A held-out Leiden sweep in the same script used too
high a resolution range (it chose partitions with 53 to 78 % unassigned) and is not reported as a
comparison.

### 4.2 Diagnosis

1. **Fragmentation is a configuration artefact first.** IVF with K = min(512, n // 6) asks for about
   six items per cluster.
   - On the 3,000-item set that gives 485 clusters over 86 classes, so pair recall is capped near
     zero by construction. k-means on continuously dense data then places those Voronoi boundaries
     arbitrarily, which is why the partition depends on row order (ARI 0.306) [R].
   - [clustering_methods.md](clustering_methods.md) section 2.4 lists "stable across reruns" as an
     IVF property. That holds only for identical row order. The measured ARI contradicts it.
   - The refine step cannot repair this. Complete linkage at a fixed 0.25 inside a bucket only
     splits further. The global complete-linkage curve is dominated by average linkage at every
     recall above 0.2 [R], because complete linkage's maximum-distance rule rejects every multi-view
     class whose two views are farther apart than the threshold.
2. **Chaining comes from the neighbourhoods.**
   - With P@10 of 0.82, about one edge in five of a 10-NN graph crosses a class boundary even on 11
     easy classes. On the 3,000-item set, with people, low resolution and 86 classes, the
     production clusters of about 8 crops are already 30 % impure.
   - Union-find (K3b) and modularity Leiden follow those edges and merge. Mutual kNN, CPM and
     CHAMELEON reduce the damage but cannot create purity the neighbourhoods do not have. That is
     why the real-data Leiden pass "only trades recall for precision": it moves along the same
     precision-recall frontier as average linkage, and does not move the frontier.
3. **Hubs are the bridges.** A crop that sits in 89 other crops' 10-NN lists (expected 10) is a
   generic vector: blurred, small, or mostly background. Hub reduction improves P@1 and P@10 a
   little and k-reciprocal helps average linkage at high recall. Both match the literature
   (section 5.4): scaling-type methods help consistently, centering only sometimes.
4. **Anisotropy breaks fixed thresholds across projects.**
   - Set P's crops are on average 0.13 more cosine-similar to each other than COCO's, and the
     effective rank is about 100. The 0.25 distance in refine, 0.7 in `POST /persons/cluster` and
     any CPM gamma therefore mean different things per project and per embedding model.
   - This matches the Leiden prototype's finding that gamma and the RI and RC floors must be
     calibrated per distribution, and the cross-domain note's T8 (calibrate per space and level).
   - Whitening is the cheapest way to make one threshold travel. It raised cluster-level precision
     on held-out halves, at some cost in recall in one fold [R].
5. **Input quality limits the embedding, before the model does.** Three effects stack:
   - center-crop truncation of elongated boxes (P@10 0.74 against 0.84)
   - upscaling of crops under 32 px by 10x or more (P@10 0.78 against 0.87 at 96 px and above)
   - no context margin
   These are measured on 544 crops, and aspect is confounded with class. They are the cheapest
   hypotheses to test because the fix is a preprocessing change and a re-encode, not a new model
   (E2).
6. **Viewpoint modes.** A class seen from the front and from the side forms two blobs. Average
   linkage and CPM can bridge them only if enough cross-view neighbour pairs exist.
   - The cross-domain note's multi-prototype idea (T1) is the right representation for this.
   - We have no viewpoint labels on real data yet, so the size of this effect is unmeasured. CO3D,
     Objectron or CompCars viewpoint labels are needed (E4 in the cross-domain plan; E8 and E9
     here).
7. **Are the embeddings the limit?** At class level and recall up to about 0.3: **no**. The
   configuration is, and fixing it gives more than twice the recall at equal precision (0.20 against
   0.088). Above that,
   and for small, elongated or person crops: **yes, as fed**. Post-hoc transforms buy 0.03 to 0.10
   precision at fixed recall. Further gains need input fixes (E2), a different frozen model (E3) or
   a trained adapter (E4). None of these is measured yet. For identity (K3, and K4 as far as it
   approaches identity), a general semantic encoder is the wrong space by design.

## 5. Sources by family

Each entry gives the method, the numbers it reports, where it applies (K1 to K4), and its evidence
tag. Entries already covered by an internal note give only the delta.

### 5.1 Internal notes: what they say, and where they agree or conflict with the measurements

| Note | Claim | Real-data status |
|---|---|---|
| Owner literature notes (untracked file `docs/cluster_research_9oct.md` in the main checkout) | Build a global refinement engine: cross-cluster kNN, mutual edges, Leiden CPM at several resolutions, CHAMELEON-style split/merge validation, stability checks, versioned hierarchy, no retraining at first; FINCH, HDBSCAN leaf, ECG and HIT-Leiden worth testing | **Partly confirmed.** Cross-cluster reconsideration is right: the fragments are real. FINCH, HDBSCAN and modularity variants failed on real crops [R]. "Leiden first choice" is not supported over plain global average linkage at low recall [R]. "No retraining first" is right for self-training (confirmation bias), but input fixes and frozen-model swaps are cheaper than graph machinery and come first |
| [clustering_methods.md](clustering_methods.md) | IVF default: balanced, streaming, stable; AHC complete 0.25 is "the strongest guard against bleed-over"; UMAP and HDBSCAN unusable | UMAP and HDBSCAN: **agrees** [R]. IVF stability: **conflicts** (ARI 0.306 under permutation). Complete linkage as the best guard: **conflicts**; average linkage dominates it on real crops (4.1). IVF as a streaming index: still valid, as an index rather than as clusters |
| [gpu_clustering.md](gpu_clustering.md) | cuML AHC is single-linkage only, has no threshold and needs dense connectivity; sklearn AHC stays on CPU | **Agrees**, and it limits the GPU path. Average linkage at scale needs a graph-HAC algorithm (5.3), not cuML |
| [cluster_types.md](cluster_types.md) | Three kinds, separate spaces, never mixed; K3b chains; 2b runs on demand | **Agrees.** Adds: 2a and 2b are mis-scaled in opposite directions (section 2) |
| Cross-domain lessons from the sibling speech project (branch `research/v080-cross-domain-clustering`, `docs/research/cross_domain_clustering_lessons.md`) | Single linkage causes false merges, so replace union-find; multi-prototype clusters; same-image cannot-link at identity level only; calibrate per space; quality gating; provenance; complete linkage at 0.75 replaced union-find in the speech project | **Agrees** on union-find, calibration and gating, and the gating direction is supported (low-res P@10 0.78) [R]. **Refines** linkage: for class-level crops, average linkage beats complete linkage [R]. Complete linkage may still be right for identity (K3), to be tested in E6. Multi-prototype: untested on real data |
| Leiden prototype (branch `research/v080-342-leiden-prototype`) | Conditional go: CPM, mutual kNN, CHAMELEON filter; calibrate per distribution | **Agrees** that real-data gains are in-sample and small. This note adds that average linkage on the same graph matches it, and that whitening moves both methods' frontier |
| Eval baseline (main and the #341 branch) | Precision 0.696, false-split rate 0.978, low-res worst | The reference numbers in 3.1 |

### 5.2 Existing applications

| App | Method (public description) | Reported numbers | Applies to | Tag |
|---|---|---|---|---|
| **Immich** (self-hosted photos) | DBSCAN-derived and incremental. Per face: search up to `minFaces` nearest faces within `maxDistance`. If any already belongs to a person, join the nearest such person. If at least `minFaces` matches exist (a core point), create a person. Otherwise defer, and on the deferred pass assign or leave as an outlier. A nightly job re-tries unassigned faces. Defaults: model `buffalo_l`, `minScore` 0.7 (detection), `maxDistance` 0.5 (cosine distance), `minFaces` 3. CLIP duplicate detection uses `maxDistance` 0.01 | No accuracy figures published | K3 directly: **same ArcFace weights as ours**, so its thresholds are a valid starting point. K1 near-duplicates (CLIP distance 0.01) | [L] docs https://docs.immich.app/features/facial-recognition ; defaults read from the repository (`e2e/src/ui/mock-network/config-fixtures.ts`, `server/src/services/person.service.ts`, https://github.com/immich-app/immich) |
| **PhotoPrism** | DBSCAN over face embeddings. `FACE_CLUSTER_DIST` (epsilon; default 0.64 in older docs, 0.72 in current docs, "calibrated per face model when unset"), `FACE_CLUSTER_CORE` (min points; 5 in current docs, 4 in older docs), a retry pass at core 4 for faces left unclustered, `FACE_MATCH_DIST` 0.25 | None. The docs warn that thresholds do not transfer between models | K3 | [L] https://docs.photoprism.app/developer-guide/vision/face-recognition/ , https://docs.photoprism.app/getting-started/config-options/ |
| **Apple Photos** | On-device face and upper-body detection, two embeddings (face, upper body). A gallery of known people is built progressively and new observations are assigned or left unassigned; upper body and capture time help when the face is turned away | The article has figures we did not read in full | K3: the second (body) embedding and the "leave unassigned" state | [L] partly. https://machinelearning.apple.com/research/recognizing-people-photos (2021); clustering details [U] |
| **Google Photos** | The product algorithm is not public. The FaceNet paper shows agglomerative clustering of a personal photo collection with its embeddings | FaceNet LFW 99.63 % [U] | K3 background only | [U] https://arxiv.org/abs/1503.03832 |
| **Sibling speech app** (speaker linking) | See 5.1: AHC seed, then VBx refinement within a recording; complete-linkage AHC at 0.75 across recordings; cannot-link, blacklist, provenance | Cross-recording linking never measured | All kinds: process lessons | [L] via the cross-domain note |
| **ortloom** (sibling Triton benchmarks) | Serving and throughput lessons only; no clustering | | none | [L] internal |

The common pattern across production photo apps:

- a **core-point rule** (a face starts a person only with at least 3 to 5 close neighbours)
- an **incremental, nearest-person assignment** with a distance cap
- an **outlier state**
- **thresholds tied to one model**

None publishes accuracy. All are DBSCAN relatives, which work for faces because an identity-trained
embedding does have density gaps. PE-Core crop space does not (HDBSCAN fails there [R]).

### 5.3 Graph, linkage and partition methods

| Method | What it adds | Reported | Fit | Tag |
|---|---|---|---|---|
| Leiden CPM (Traag et al. 2019) | Well-connected communities, resolution knob, initial partition (`leidenalg`). Signed graphs through multiplex layers with weight -1, which is a way to encode cannot-link | | K2, K3 | [L] https://www.nature.com/articles/s41598-019-41695-z ; negative links: https://leidenalg.readthedocs.io/en/stable/multiplex.html |
| CHAMELEON (Karypis et al. 1999) | Merge test by relative interconnectivity and closeness | | K2 merge validator | [L] https://doi.org/10.1109/2.781637 |
| ECG (Poulin and Theberge 2018) | Ensemble of level-1 Louvain runs; available in cuGraph | Real data: none. Synthetic scale only (3.2) | K2 GPU candidate | [L] https://arxiv.org/abs/1809.05578 |
| FINCH, HDBSCAN, UMAP + clustering, modularity Leiden | | Rejected on real crops (3.1) | | [R] |
| **Approximate rank-order clustering** (Otto, Wang, Jain, TPAMI 2018) | Distance from shared positions in each other's approximate kNN lists, then a simplified merge. A per-cluster quality score picks clusters for manual review | 123M face images into more than 10M clusters; more accurate than k-means and spectral clustering on their benchmarks | K3 at millions; the per-cluster quality score suits the review queue | [L] https://arxiv.org/abs/1604.00989 |
| **Graph HAC at scale: SCC, ParHAC, TeraHAC** (Google) | Average-linkage-style HAC on a sparse similarity graph, (1+epsilon)-approximate, distributed | TeraHAC: graphs up to 8 trillion edges, up to 8.3x faster than SCC with 1.16x higher quality | **K2 at millions**: the scalable form of the average linkage that won in 4.1. sklearn cannot do it (memory), and neither can cuML (single linkage only) | [L] https://arxiv.org/abs/2308.03578 ; SCC: Monath et al. KDD 2021 https://arxiv.org/abs/2010.11821 [U: venue not re-read] |
| **Hierarchical k-means for balanced curation** (Vo et al. 2024, Meta) | k-means applied repeatedly and hierarchically, then balanced sampling across the tree; used to curate DINOv2-style pre-training data | Curated data matches or beats manually curated data for SSL; 84.7 % ImageNet top-1 per a secondary summary [U] | K1 (browsable, balanced topic tree over millions of images), and the auto-label loop's sampling | [L] https://arxiv.org/abs/2405.15613 , code https://github.com/facebookresearch/ssl-data-curation |
| HIT-Leiden (2026), dynamic Leiden | Incremental maintenance of Leiden communities | Benchmark speed-ups only | K2 continuous refinement, later | [L] via the owner notes and the cross-domain note |

### 5.4 Embedding-space corrections (no training)

| Method | What it does | Reported | Our real result | Tag |
|---|---|---|---|---|
| Hubness (Radovanovic et al. JMLR 2010) | In high dimensions some points appear in many kNN lists; hubs degrade kNN methods | | Skew 2.4 to 2.6, max N10 89 to 119 [R] | [L] https://jmlr.org/papers/v11/radovanovic10a.html [U: URL not opened] |
| Hubness reduction compared (Feldbauer and Flexer, KAIS 2019) | 12 methods on 50 datasets: **scaling methods (local scaling, mutual proximity) consistently improve hubness and kNN accuracy; centering only in specific settings** | | Consistent with ours: centering no gain; CSLS and k-reciprocal reduce hubness [R] | [L] https://link.springer.com/article/10.1007/s10115-018-1205-y ; `scikit-hubness` https://arxiv.org/abs/1912.00706 |
| Local and global scaling, mutual proximity (Schnitzer et al. JMLR 2012) | Rescale each distance by the neighbourhood density of both points | | Not run (CSLS is the local-scaling arm) | [L] https://dl.acm.org/doi/10.5555/2503308.2503333 |
| CSLS (Conneau et al. ICLR 2018) | 2cos - r(x) - r(y) with r the mean similarity to the k nearest reference points | | Hub skew to 1.63, P@10 -0.01 [R] | [U] https://arxiv.org/abs/1710.04087 |
| PCA whitening (Jegou and Chum, ECCV 2012) | Down-weights dominant, co-occurring directions; standard in image retrieval | | Avg-AHC precision +0.03 to +0.11 at matched recall, held-out precision higher in 6 of 6 cells, recall lower in 2 [R] | [U] Jegou and Chum 2012 (URL not checked) |
| k-reciprocal re-ranking (Zhong et al. CVPR 2017) | Jaccard distance over expanded k-reciprocal sets | | Hub skew 2.57 to 0.69, P@1 +0.026, best avg-AHC precision at R >= 0.5 (0.752) [R] | [L] https://arxiv.org/abs/1701.08398 |
| Query expansion / DBA (Radenovic et al. TPAMI 2018); diffusion (Iscen et al. CVPR 2017) | Replace a vector by a weighted mean of its neighbours, or diffuse similarity on the kNN graph | | DBA: P@1 +0.028, no clustering gain [R]; diffusion not run | [U] https://arxiv.org/abs/1711.02512 , https://arxiv.org/abs/1611.05113 |
| Adaptive s-norm (Matejka et al. 2017) | Normalise each score against each item's top impostors | | Not run (cross-domain note T8) | [L] via the cross-domain note |

### 5.5 Face clustering

| Method | Idea | Reported (pairwise F_P / BCubed F_B) | Training need | Tag |
|---|---|---|---|---|
| L-GCN (Wang et al. CVPR 2019) | GCN predicts linkage on each node's sub-graph | | MS1M-style labelled faces | [U] https://arxiv.org/abs/1903.11306 |
| GCN-V / GCN-E (Yang et al. CVPR 2019, 2020) | Vertex confidence plus edge connectivity on the affinity graph | F_P 79.3 on the 5.21M MS1M split (as quoted by STAR-FC) | MS1M | [L] https://arxiv.org/abs/2004.00445 |
| STAR-FC (Shen et al. CVPR 2021) | Structure-preserving subgraph sampling; trains on 20M nodes, infers on 12M | F_P 91.97 on the 584K MS1M split in 310 s; F_P 83.46 / F_B 81.47 on 5.21M | MS1M | [L] https://arxiv.org/abs/2103.13225 |
| Ada-NETS (Wang et al. ICLR 2022) | Learns which neighbours to keep in a "structure space" before the GCN | F_B about 91.4, F_P about 92.7 on 584K (official README) | MS1M | [L] https://arxiv.org/abs/2202.03800 , https://github.com/damo-cv/Ada-NETS |
| Hi-LANDER (Xing et al. ICCV 2021) | Hierarchical GNN: linkage plus density per level, merging connected components level by level; **also tested on non-face image sets** | F-score 0.224 to 0.700 on the Hannah set; about 49 to 54 % average F gain over earlier GNN methods | labelled training identities disjoint from the test set | [L] https://arxiv.org/abs/2107.01319 , code in DGL `examples/pytorch/hilander` |
| Rank-order (Otto et al.) | See 5.3 | 123M faces | none | [L] |

Licence caveat: the learned-linkage papers train and test on MS-Celeb-1M splits. Microsoft withdrew
MS-Celeb-1M in 2019 [U: withdrawal date from memory]. Our ArcFace weights are trained on WebFace600K
under InsightFace's non-commercial terms (`ATTRIBUTION.md`). A learned linkage model would need a
licence-clean training set, and that is an owner decision. The training-free core-point family
(Immich, PhotoPrism, rank-order) avoids it.

### 5.6 Deep and foundation-model clustering

| Method | Idea | Reported | Fit | Tag |
|---|---|---|---|---|
| DeepCluster, SwAV | Alternate clustering and representation learning | | Owner notes: confirmation-bias risk; not first | [L] via the owner notes |
| SCAN (Van Gansbeke et al. ECCV 2020) | Mine kNN from a pretext embedding, train a clustering head for neighbour consistency, then self-label | | K2 head on frozen PE features, later | [U] https://arxiv.org/abs/2005.12320 |
| TEMI (Adaloglou et al. BMVC 2023) | Clustering heads on frozen pretrained features with a mutual-information objective over kNN pairs | | K2, K1 | [U] https://arxiv.org/abs/2303.17896 |
| TURTLE (Gadetsky et al. ICML 2024) | Searches for the labelling that yields maximum-margin linear classifiers in one or more frozen foundation-model spaces | Matches CLIP zero-shot on average over 26 datasets with one space; beats it with two | K2 or K4 with a fixed K and several frozen spaces (PE plus DINOv2) | [L] https://arxiv.org/abs/2406.07236 |

These methods assume a known cluster count and roughly balanced classes. Curation pools are
long-tailed with unknown K, so the methods are candidates for a per-class head (K4), not for the
global pool.

### 5.7 Re-identification and multi-view

Covered in the cross-domain note (VANet, PVEN, TransReID, CLIP-ReID, SpCL, cluster contrast, ISE,
PPLR, k-reciprocal). Its key figure: identity-trained models reach about 80 to 85 mAP on VeRi-776
[L]. The open question it leaves, whether frozen PE-Core links a front view to a rear view at all,
is still unmeasured. Nothing new is added here, except that multi-view structure matters for K2 at
class level too (4.2 item 6).

### 5.8 Fine-grained and open-set recognition (K4)

| Source | Point | Reported | Tag |
|---|---|---|---|
| GCD (Vaze et al. CVPR 2022) | Generalized category discovery: cluster unlabelled data that holds both known and novel classes, using a labelled subset | Stanford Cars "All" 39.0 (as quoted by SimGCD) | [L] https://arxiv.org/abs/2201.02609 |
| SimGCD (Wen et al. ICCV 2023) | Parametric classifier baseline for GCD | **Stanford Cars 53.8 All / 71.9 Old / 45.0 New**, DINO ViT-B/16, true K = 196; 49.1 / 65.1 / 41.3 with an estimated K | [L] https://arxiv.org/abs/2211.11727 , https://github.com/CVMI-Lab/SimGCD |
| Open-set recognition (Vaze et al. ICLR 2022) | A good closed-set classifier, plus a max-logit score, is a strong unknown detector | | [U] https://arxiv.org/abs/2110.06207 |
| Label propagation (Zhou et al. 2004; Iscen et al. CVPR 2019) | Diffuse a few labels over the kNN graph | | [U] https://arxiv.org/abs/1904.04717 |
| Datasets | Stanford Cars: 196 make/model/year classes, about 16k images; licence terms to confirm (original host offline). CompCars: viewpoint labels, non-commercial. VMMRdb: thousands of make/model classes [U] | | [U] for the counts not re-read |

Reading for K4: unsupervised fine-grained clustering with general features is weak even in the best
papers. A trained GCD method with half the classes labelled gets about half the Cars images right.
K4 should be built as "few seeds, then propagation, then an unknown bucket", not as discovery from
scratch.

### 5.9 Constrained, semi-supervised and active clustering

Covered in the cross-domain note: COP-KMeans, confidence-weighted constraints, A3S active queries,
pseudo-label confirmation bias. Additions:

- leidenalg can carry cannot-links as a negative-weight layer (5.3).
- **The detector class is a free soft constraint.** It agrees with ground truth on 92 % of COCO
  crops [R]. Partitioning within the detector class, or a cannot-link between confident different
  detector classes, is a K2 prior that needs no human. It does not apply to the unconfident residual
  items it is least sure about. Measure it on that subset (E5).
- **The same-image cannot-link applies at identity level only** (K3, and K4 if K4 becomes
  identity). It must stay out of class-level graphs, as the cross-domain note says.

### 5.10 VLM- and LLM-assisted clustering and judging

| Source | Point | Tag |
|---|---|---|
| IC|TC (Kwon et al. ICLR 2024) | A VLM captions each image, an LLM proposes and assigns clusters under a user's text criterion, and the user iterates. Critics note it is text clustering grounded in captions | [L] https://arxiv.org/abs/2310.18297 , https://github.com/sehyunkwon/ICTC |
| VLM as judge | No paper was found that validates VLM judgements of cluster purity against human labels for object crops. Treat judge agreement as something we must measure (E0) | [U] |

The visual-verification idea, as a protocol (section 8, E0):

- Show a VLM (a hosted small model such as Claude Haiku, or the local catalog VLM) a grid of
  sampled members of one cluster and ask which members do not belong.
- Show pairs and ask "same class?" (K2), "same make and model?" (K4) or "same object?" (identity).
- Use it only after its agreement with ground truth is measured on public labelled crops.
- **Never use it for face identity on private photos.** That is biometric and outside the judge's
  reliable competence.

### 5.11 GPU and scale

Covered by [gpu_clustering.md](gpu_clustering.md), the Leiden prototype and the cross-domain note:

- FAISS GPU and cuVS CAGRA are the kNN builders.
- cuGraph Leiden is modularity-only, with no initial partition. ECG is the GPU community option.
- cuML AHC is single linkage only.

Additions:

- At 1024-d, kNN dominates cost (3.2). PCA to 256 dimensions, which section 4 shows helps quality,
  also cuts kNN cost by about 4x [S: arithmetic, not measured].
- Average linkage at millions needs graph HAC (TeraHAC/SCC-style, 5.3). No GPU implementation was
  found [U].

## 6. Comparison across methods

Real-data results are on the COCO project unless marked. "In-sample" means the parameter was chosen
on the scored labels.

| Method | Kinds | Literature figure | Our real-data result | Synthetic (scale only) | Verdict |
|---|---|---|---|---|---|
| FAISS IVF, K = n // 6 (current 2a) | K2, K1, K3a | | 3,000 set P 0.696 / R 0.022; COCO 0.905 / 0.088; ARI 0.306 under row permutation | 1M in 596 s | Keep as a coarse **index** and streaming assigner. Stop treating cells as clusters |
| MiniBatchKMeans, n / 800 (2b) | K2b | | not scored | | Score it (E1) |
| Complete-linkage AHC 0.25 (refine) | K2 | | Global curve 0.878 at R >= 0.2, 0.730 at 0.3 | | Dominated by average linkage for class level |
| **Average-linkage AHC, global** | K2 (K4 within class) | | 0.920 at R >= 0.2 (in-sample curve); held-out 0.796 to 0.958 at R 0.11 to 0.37 | | **Candidate default**, via graph HAC at scale |
| **+ PCA whitening 256** | K2, K4 | | 0.962 at 0.2, 0.906 at 0.3; held-out 0.891 to 0.976; precision higher in 6 of 6 cells, recall lower in 2 | | **Candidate default transform**; confirm on the 3,000 set (E1) |
| Leiden CPM, mutual 30-NN | K2, K3 | | 0.925 at 0.2, 0.889 at 0.3, **0.776 at 0.5**; prototype in-sample 0.905 / 0.234 | 500k 1024-d in 380 to 450 s | Keep as the high-recall arm; not better at low recall |
| Leiden + CHAMELEON | K2 | | 0.944 / 0.195 in-sample, 9.8 % unassigned | filter slow on many atoms | Merge validator only if it beats average linkage held-out |
| Consensus Leiden, ECG | K2 | | consensus 0.758 / 0.526 (no gain); ECG not run on real data | ECG 20k in 7 s | ECG: run once on real data (E1); consensus: drop |
| Modularity Leiden (incl. cuGraph) | | | 0.709 / 0.579 | | Reject |
| HDBSCAN, FINCH, UMAP + any | | | rejected (3.1) | | Reject for K2. HDBSCAN-like core points remain valid for K3 |
| k-reciprocal re-rank | K2, K3 | Zhong 2017 | hub skew 0.69; avg-AHC 0.752 at R >= 0.5 | | High-recall arm; test with Leiden on the 3,000 set |
| CSLS, centering, DBA | | Feldbauer and Flexer: centering weak | no clustering gain | | Drop centering and DBA; CSLS optional |
| Union-find 0.7 (current K3b) | K3 | chaining (cross-domain note) | not scored | dense n^2: 40 GB at 100k faces | Replace (E6) |
| Immich core-point incremental | K3 | no published accuracy | not scored | | **First K3 baseline** (same weights) |
| PhotoPrism DBSCAN | K3 | none | not scored | | Second K3 baseline |
| Rank-order (Otto) | K3 | 123M faces | not scored | | Scale arm for K3 |
| STAR-FC / Ada-NETS / Hi-LANDER | K3 (Hi-LANDER also K2) | F_P 91.97, F_B 91.4 (MS1M 584K) | not scored | | Only if training-free methods fall short and a clean training set exists |
| Hierarchical k-means (Vo 2024) | K1 | SSL curation | not scored | | K1 browsing tree |
| SimGCD / GCD | K4 | Cars 53.8 All | not scored | | Shows K4 needs seeds |
| TURTLE / TEMI / SCAN | K2 heads, K4 | TURTLE: CLIP zero-shot parity | not scored | | Later; known-K assumption |
| Label propagation | K4, K2 | | not scored | | K4 core (E9) |
| VLM judge | all | none found | not measured | | Measure agreement first (E0) |

## 7. Recommended stack per kind

The recommendations rest on 544 real labelled crops and the literature. Each step names the
experiment that confirms or kills it.

### K2: detection crops and items (class level, any class, multi-view)

1. **Input hygiene first.**
   - Letterbox or add a context margin for the PE crop instead of a center-crop (E2).
   - A quality gate: crops under 32 px or blurred join clusters but never form or bridge one
     (cross-domain T9).
2. **One transform per embedding model and project:** PCA to 256 dimensions plus whitening, fitted
   on the project's own pool or a reference pool, then re-normalised. Store it with the partition
   run (E1).
3. **Coarse index: keep IVF,** but only for streaming assignment and kNN candidate search. K follows
   index needs, not cluster needs.
4. **Partition: sparse kNN graph (k 30, mutual), then average-linkage graph HAC.**
   - The cut is calibrated on a dev split: the largest recall whose dev precision is at least the
     target.
   - Leiden CPM is the comparison arm. Keep whichever wins held out at the operating point the
     review queue needs.
   - At millions, use TeraHAC/SCC-style approximate HAC or Leiden. Plain sklearn is not an option.
5. **Priors and validation.**
   - The detector class as a soft prior (E5).
   - CHAMELEON RI and RC as a merge validator only if it adds precision held out.
   - Abstention: items with a low margin go to review.
6. **Multi-prototype clusters** (cross-domain T1) for view modes. Measure on viewpoint-labelled data
   (E8).
7. **Representation upgrade, only if steps 1 to 4 plateau.** A frozen alternative (E3), then a small
   adapter trained on public labelled crops and checked on unseen classes (E4).

### K3: faces (identity)

1. Quality gate: detection score at least 0.7 (Immich default), a minimum face size, optionally a
   blur check.
2. **Baseline: Immich-style core-point incremental assignment** with the same `buffalo_l` weights:
   distance 0.5, core 3. The nightly re-try handles deferred faces.
3. Batch refinement on the kNN graph, never dense n^2: average or complete linkage graph HAC, or
   Leiden CPM, plus k-reciprocal weights. Add the same-image cannot-link and the named persons and
   merges as hard anchors.
4. Multi-prototype persons (pose modes), and the stable-id rule of #343 so names survive a re-run.
5. Learned linkage (STAR-FC, Ada-NETS, Hi-LANDER) only on evidence and with a licence-clean
   training set.

### K4: fine-grained groups (e.g. make/model)

1. Run K2 first, to get a reliable "car" class.
2. Within the class: measure first whether PE-Core separates make/model at all (P@10 on Stanford
   Cars, E9). Compare frozen DINOv2 and SigLIP 2.
3. Seeds: a few labelled examples per make/model (human or imported), or names from the VLM on
   representative medoids, verified by the judge.
4. Label propagation on the within-class kNN graph, an open-set score, and an explicit unknown
   bucket. Never force-assign.
5. A per-class adapter or head (supervised contrastive, or a GCD-style parametric head) once enough
   seeds exist. The multi-viewpoint problem is solved with multi-prototype classes.

### K1: whole image

1. Near-duplicates: a tight CLIP threshold. Immich uses 0.01 cosine distance; our existing
   `duplicate_group_id` mechanism is separate (see [cluster_types.md](cluster_types.md)).
2. Browsing: a hierarchical k-means tree (Vo et al. 2024) over MobileCLIP vectors, plus metadata
   (time, place) for events. Cells are navigation, not identity.
3. Score it with image-level labels (COCO and Open Images multi-label: Jaccard of label sets as
   relevance) and the VLM judge (E7).

## 8. Experiments, real data only, with a held-out split

**Split rules (all experiments).**

- Split by image, never by crop, so crops of one image never sit on both sides. Stratify by class.
- 50 % dev and 50 % test. Every threshold, gamma, floor and transform choice is made on dev. Test is
  read once per method family.
- Report test precision and recall with 1,000-resample bootstrap intervals over images.
- Fit transforms on data outside the scored set: the project pool or set P.
- For K4 and for the open-set check, also use a class-disjoint split: calibrate on one set of
  classes and test on unseen classes (GCD protocol).
- Set P has no ground truth. It is scored only by the VLM judge (after E0), by stability, and by
  owner spot checks. Committed text carries its numbers only.

| ID | Experiment | Data | Measures | Confirms / kills | Cost |
|---|---|---|---|---|---|
| E0 | **VLM judge calibration.** 200 sampled clusters (grids of 9 to 16 crops, stratified by size and purity) and 500 pairs, judged by a hosted small model (Haiku) and the local catalog VLM | 3,000-item set (labels known) | Agreement with ground truth: Cohen's kappa on pairs; precision and recall of the "does not belong" flags; cost per judgement | Use the judge on set P only if pair kappa is at least 0.7 and flag precision at least 0.8 (proposed bars, to be agreed with the owner). Otherwise the judge only ranks clusters for human review | Hours; API budget, or GPU 0 |
| E1 | **Reproduce section 4 at 3,000 items.** IVF (current), global avg-AHC, complete-AHC, Leiden CPM, ECG once; raw, whitened 256 and k-reciprocal; also score the 2b MiniBatchKMeans partition | Re-ingest the pinned 3,000-item set into a throw-away project (`-p` name) | Test-fold precision and recall per stratum, unassigned share, ARI under row permutation | **Kills whitening** if the test precision gain at matched recall is under 0.02. **Kills Leiden as default** if it does not beat avg-AHC at the chosen recall | Under 1 h CPU after ingest |
| E2 | **Crop preprocessing.** Re-encode the same crops (a) as now, (b) letterboxed to square, (c) with a 10 % and a 20 % context margin | 3,000-item set; PE-Core on GPU 0 under the lock | P@10 per aspect and size stratum; E1 curves | **Confirms** if elongated-crop P@10 rises by at least 0.03 with no loss elsewhere; otherwise keep the center-crop | About 1 GPU-hour |
| E3 | **Frozen embedding alternatives.** DINOv2-L, SigLIP 2 (L or So400m), PE-Core-L (current), and PE-Core-G if it fits | 3,000-item set crops; Stanford Cars test for K4 | P@10, 10-NN vote, E1 curves per stratum (person, low-res) | A model must beat PE-Core by at least 0.03 P@10 on the low-res and person strata to justify a second vector; otherwise PE stays | A few GPU-hours; torch in a venv or Docker |
| E4 | **Adapter.** A linear or 2-layer head on frozen PE (supervised contrastive) trained on COCO train2017 crops; tested on COCO val crops **and on Open Images classes absent from COCO** | COCO train / val, Open Images | Gain on seen classes against unseen classes | **Kills** the adapter if unseen-class P@10 drops (overfitting to COCO's 80 classes) | 1 to 2 days |
| E5 | **Detector-class prior.** Partition within the detector class against globally, on the low-confidence subset only | 3,000-item set | Precision and recall on items under the residual confidence cut | Adopt if precision rises at equal recall on the residual-like subset | 1 h |
| E6 | **Faces.** Union-find 0.7 against the Immich algorithm (0.5, core 3, score 0.7) against PhotoPrism-style DBSCAN against avg and complete graph HAC and Leiden CPM, with and without the same-image cannot-link and k-reciprocal weights | LFW (13k images, 5.7k people, public); a cross-pose set (CPLFW or CFP-FP, licence to verify); faces detected on COCO person crops for multi-face scenes; SCRFD plus ArcFace through the stack | BCubed and pairwise F, chaining incidents (clusters with more than one identity), split rate per identity size, memory at O(n k) | Choose the K3 replacement. Union-find stays only if nothing beats it | 1 to 2 days |
| E7 | **Whole image (K1).** Hierarchical k-means against flat IVF against avg-HAC on MobileCLIP | COCO val2017 and the Open Images val sample (image-level labels) | Label-set Jaccard purity, judge-rated coherence (after E0) | Pick the K1 browsing structure | 1 day |
| E8 | **Viewpoint.** Multi-prototype against centroid assignment; cross-view pair recall | CO3D or Objectron (licence check), CompCars viewpoint labels (non-commercial, local only) | Same-class (and same-object) pair recall per azimuth bin at a fixed false-merge rate | Whether multi-prototype clusters are needed for K2 and K4 | 2 to 3 days |
| E9 | **Fine-grained (K4).** PE-Core, DINOv2 and SigLIP 2 on Stanford Cars test (196 classes): kNN P@10; avg-HAC curves; label propagation from 1, 3 and 5 seeds per class; open-set split 98 known / 98 novel; VLM naming accuracy on 200 medoids | Stanford Cars (licence to confirm), then CompCars or VMMRdb | Accuracy at coverage; unknown-detection AUROC | Shows whether K4 is feasible with frozen features plus seeds, or needs an adapter | 2 days |
| E10 | **Scale on real vectors.** kNN (FAISS GPU or CAGRA, PCA 256 against 1024), graph HAC and Leiden at 90k (set P) and about 860k (COCO train2017 annotated objects, public and real) | set P; COCO train2017 crops | Wall time, memory, kNN recall; same precision checks on COCO train labels | The real-data replacement for the synthetic 1M runs | 1 day plus an ingest of about 118k images |

Order: E0 and E1 first, because they decide whether everything else is measured with the judge
and against which default. E2 and E5 are the cheapest likely wins. E6 is independent and can run in
parallel. E3, E4, E8, E9 and E10 follow.

## 9. What evidence is missing (honest list)

1. **Sample size.** Section 4 rests on 544 labelled crops of 11 common classes from 200 images. The
   held-out folds disagree by up to 0.2 recall. Nothing here is a final value. E1 is the
   confirmation.
2. **No real-data score at all** for K1, K2b, K3 (either mechanism) or K4.
3. **No viewpoint-labelled real data.** "Viewpoint modes cause fragmentation" is a hypothesis.
4. **The preprocessing effect is inferred, not measured.** Elongated crops are worse, but aspect is
   confounded with class (bus, train, person). Only a re-encode (E2) separates the two.
5. **Alternative embeddings unmeasured:** DINOv2, SigLIP 2 and PE variants on our crops. PE-Core's
   fine-grained (Cars) performance was not found in the PE paper's accessible text [U].
6. **VLM judge validity unmeasured** (E0). No published validation for object-crop cluster judging
   was found.
7. **Residual-pool realism.** The labelled sets score all crops. Production clusters only
   unconfident items, which are harder. The detector-prior and precision numbers will be lower
   there.
8. **Scale quality.** Every large-n run so far is synthetic or unlabelled. E10 uses COCO train2017
   (about 860k real annotated objects) to close that gap.
9. **Licences.** MS-Celeb-1M (withdrawn), Stanford Cars, CompCars, VMMRdb, CO3D, IJB-B, CPLFW and
   CFP-FP all need a licence check before use; some may be local-only. This is an owner decision.
10. **Reproducibility of section 4.** The diagnostic scripts are research scratch code outside the
    repository (Reproduce, below). They should land with the #342 research scripts before any number
    from section 4 is quoted in the paper.

## 10. Reproduce section 4

Setup: the research CPU venv of the Leiden prototype (numpy, scipy, scikit-learn, faiss-cpu 1.15.1,
igraph, leidenalg 0.12.0, httpx). The test stack's OpenSearch must hold the public COCO project, and
the labels and `coco_gt.json` come from the cluster-eval set build. Scratch scripts (not in the
repository): `fetch.py`, `diag.py` and `heldout.py` in the author's scratch directory
(not committed; see section 9 item 10).

```bash
PY=$SCRATCH/venv-cluster-research/bin/python
# read-only export: 1,364 COCO crops, and 40,000 set P crops for fitting transforms
$PY -I fetch.py http://127.0.0.1:5607 <coco project items index> coco 100000
$PY -I fetch.py http://127.0.0.1:5607 <set P project items index> setp 40000
OMP_NUM_THREADS=6 nice -n 19 $PY -I diag.py      # writes diag.json and curves.json
OMP_NUM_THREADS=6 nice -n 19 $PY -I heldout.py   # image-half held-out check
```

Definitions used:

- P@k counts neighbours among the labelled crops only.
- Pair precision and recall are computed as in `scripts/bench/cluster_eval_metrics.py`, over the
  labelled crops.
- Clusters under 3 members are unassigned for Leiden only.
- Hubness is the skewness of the 10-occurrence counts over all 1,364 crops.
- The effective rank is exp(entropy of the normalised PCA spectrum).
- Whitening divides by the square root of the eigenvalue after projecting onto the top 256 set P
  principal components, then re-normalises.
- k-reciprocal follows Zhong et al. 2017 (k1 20, k2 6, lambda 0.3), computed inside the 1,364 crops.

Raw outputs: `diag.json`, `curves.json` and `diag.log` in the same scratch directory.

## 11. References

Internal:

- [cluster_types.md](cluster_types.md), [clustering_methods.md](clustering_methods.md),
  [gpu_clustering.md](gpu_clustering.md), [cluster_eval_baseline.md](cluster_eval_baseline.md),
  [v070_measurement_log.md](v070_measurement_log.md),
  [embedding_model_eval_plan.md](embedding_model_eval_plan.md)
- Unmerged branches: `research/v080-342-leiden-prototype` (Leiden prototype, `fe29152c`),
  `research/v080-cross-domain-clustering` (cross-domain lessons, `ceb92674`), `fix/v070-341-work`
  (3,000-item baseline, `6c58acbd`)
- Untracked owner notes: `docs/cluster_research_9oct.md` (main checkout)

External, by section:

- 5.2 Immich https://docs.immich.app/features/facial-recognition , https://github.com/immich-app/immich ;
  PhotoPrism https://docs.photoprism.app/developer-guide/vision/face-recognition/ ,
  https://docs.photoprism.app/getting-started/config-options/ ; Apple
  https://machinelearning.apple.com/research/recognizing-people-photos ; FaceNet
  https://arxiv.org/abs/1503.03832
- 5.3 Leiden https://www.nature.com/articles/s41598-019-41695-z ; leidenalg multiplex and negative
  links https://leidenalg.readthedocs.io/en/stable/multiplex.html ; CHAMELEON
  https://doi.org/10.1109/2.781637 ; ECG https://arxiv.org/abs/1809.05578 ; rank-order
  https://arxiv.org/abs/1604.00989 ; TeraHAC https://arxiv.org/abs/2308.03578 ; SCC
  https://arxiv.org/abs/2010.11821 ; hierarchical k-means curation https://arxiv.org/abs/2405.15613
- 5.4 Hubness https://jmlr.org/papers/v11/radovanovic10a.html ; Feldbauer and Flexer
  https://link.springer.com/article/10.1007/s10115-018-1205-y ; scikit-hubness
  https://arxiv.org/abs/1912.00706 ; Schnitzer et al. https://dl.acm.org/doi/10.5555/2503308.2503333 ;
  CSLS https://arxiv.org/abs/1710.04087 ; k-reciprocal https://arxiv.org/abs/1701.08398 ; DBA/QE
  https://arxiv.org/abs/1711.02512 ; diffusion https://arxiv.org/abs/1611.05113
- 5.5 L-GCN https://arxiv.org/abs/1903.11306 ; GCN-V/E https://arxiv.org/abs/2004.00445 ; STAR-FC
  https://arxiv.org/abs/2103.13225 ; Ada-NETS https://arxiv.org/abs/2202.03800 ; Hi-LANDER
  https://arxiv.org/abs/2107.01319
- 5.6 SCAN https://arxiv.org/abs/2005.12320 ; TEMI https://arxiv.org/abs/2303.17896 ; TURTLE
  https://arxiv.org/abs/2406.07236
- 5.8 GCD https://arxiv.org/abs/2201.02609 ; SimGCD https://arxiv.org/abs/2211.11727 ; open-set
  https://arxiv.org/abs/2110.06207 ; label propagation https://arxiv.org/abs/1904.04717
- 5.10 IC|TC https://arxiv.org/abs/2310.18297
- Re-ID, constrained clustering, speaker diarisation and GPU library references: see the
  cross-domain note's reference list (its section 11)
