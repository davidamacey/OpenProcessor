# Clustering research plan: the best grouping for curation, on the new foundation

Status: plan, research. Nothing here is implemented. Written 2026-11-04 against `5ca6c5b0`.
Depends on: program B2 (every product change lands as a Celery job); program B5 (the GPU
pipeline: decode once, crops at model size; research encodes run on that stack so numbers match
the product). Research itself starts after the v0.7.0 tag ([program index](program/00_index.md)
section 10: research is outside the program). Writing and committing the pre-registered bars
(phase R0) is docs only and may happen at any time.
Tracker: #346. Family: #398, #400, #401, #406, #416, #347, #365, #435, #436 (#402 is closed;
its remainder is #435 and #436).
Sources: [cluster_program_outcome.md](../research/cluster_program_outcome.md) (primary record),
[cluster_search_makemodel_plan.md](cluster_search_makemodel_plan.md) (KEEP-OPEN, parked waves),
[12_prior_plans_reconciliation.md](program/12_prior_plans_reconciliation.md) section 2.2,
[retrieval_baseline.md](../research/retrieval_baseline.md),
[clustering_research_consolidated.md](../research/clustering_research_consolidated.md),
[cluster_config_decision.md](../research/cluster_config_decision.md),
[cluster_config_z4.md](../research/cluster_config_z4.md),
[cluster_config_loop_cost.md](../research/cluster_config_loop_cost.md),
[leiden_refinement_prototype.md](../research/leiden_refinement_prototype.md),
[wb2_standard_benchmarks.md](../research/wb2_standard_benchmarks.md),
[visual_judge_e0.md](../research/visual_judge_e0.md),
[labelled_benchmarks.md](../research/labelled_benchmarks.md),
[gpu_clustering.md](../research/gpu_clustering.md),
[embedding_model_eval_plan.md](../research/embedding_model_eval_plan.md), the issues above with
their comments, and the library docs cited in section 3.

Audience: a fresh agent with no conversation memory. Each experiment (section 5) names its data,
arms, compute, window, bars and output. A plan agent or the coordinator turns one experiment into
a lane brief; this file is the design source.

## 0. Rules this plan is held to

| Rule | What it means here |
|---|---|
| Real data decides | Quality claims come from real photos only. Published numbers use public COCO or Open Images. Synthetic vectors may carry runtime and memory only; `scripts/bench/scorecard.py` already refuses quality fields on a synthetic dataset |
| Private set is owner-run | The owner's private photo set is an option the owner runs himself. Committed text holds aggregate numbers only; nothing per image, no paths, no names |
| Visual check | Every method that passes its machine bars is checked on contact sheets by a judge (Haiku on public non-face, non-plate images; the local VLM for faces and plates) and by a blind human re-judge of a seeded sample |
| Faces too | Faces are a first-class research line (RQ-9), at identity level, never sent to a hosted model |
| Cut unproven research | Every line has a stop rule (section 7). A line that fails its bar is parked with an evidence comment, the product keeps its current behaviour, and the outcome note records done and not done |
| Ship track first | The cheap, directly shippable questions run first (RQ-1 class roots, RQ-2 near-duplicate threshold, RQ-8 judge) |
| Agreement-gated GPU | A GPU clustering backend is adopted only when it agrees with the CPU result on real COCO data (cluster purity and label agreement within a stated tolerance). Bit-identical output is not required (owner decision of 2026-10-10) |
| Class identity by name | Ground truth, roots, exports and cross-dataset maps carry class names. A COCO-to-Open-Images name map is committed as data; no raw class index crosses a boundary |
| Multi-user correctness | No in-process or file locks for shared state. Product jobs run as Celery tasks under B1 leases; activation of a result is an OCC write |
| Libraries before hand-rolling | FAISS, cuVS, cuGraph, leidenalg, scikit-learn, statsmodels and SciPy first (section 3). Custom code only where no maintained library does the job, with the reason written down |
| Keep the benchmark data | `docs/benchmarks/cluster_eval/`, `docs/benchmarks/retrieval_baseline/`, `docs/benchmarks/wb2_standard_benchmarks/` and `docs/benchmarks/cluster_config_baseline_3000.json` are read, never changed or pruned. New results go into new folders with a `SHA256SUMS` file |
| Pre-register | Bars are committed before any score is computed; a bar moved after a read needs a new registration and a statement of why |
| No private names | No private project, runner or dataset names in any committed text |

## 1. What was learned so far

All figures are quoted from the linked notes, which hold the method and limits. Every run so far
was on a loaded host (load average 9 to 79), so only within-table timing comparisons hold.

### 1.1 Production state

| Kind | What ships | Measured quality | Source |
|---|---|---|---|
| Items (curation crops) | FAISS IVF, K = min(512, n // 6), over PE-Core-L14-336 (1024-d) | 3,000-item set (944 images, 3,961 crops, 2,950 labelled, 86 classes): held-out pair precision 0.716, pair recall 0.027, BCubed precision 0.784, BCubed recall 0.186. 539-crop COCO project: pair precision 0.905, recall 0.088 | [cluster_eval_baseline.md](../research/cluster_eval_baseline.md), `docs/benchmarks/cluster_eval/score.json` |
| Region boxes | MiniBatchKMeans k = max(8, round(n / 800)), AHC refine, on demand | no evaluation set of its own | [cluster_types.md](../research/cluster_types.md) |
| Faces (core) | `POST /persons/cluster`: exact FAISS range search, connected components at cosine 0.7 (single linkage) | never scored; about 4 min at 200k faces, about 1.5 h at 1M, inside the request | #346 comment of 2026-11-01; `src/services/face_identity/person_grouping.py` |
| Whole images (core) | FAISS IVF cells over MobileCLIP 512-d; the curation frame vector is never clustered | none | [cluster_types.md](../research/cluster_types.md) |
| Near-duplicates | core ingest 0.99 (MobileCLIP); curation frame 0.98 (PE); curation crops 0.98 (`OP_CROP_DUP_THRESHOLD`, union-find inside IVF buckets, marked "a hypothesis, not a finding" in the code) | core 0.99 catches 34 of 100 JPEG q60 copies and 13 of 100 half-size copies; 0.95 catches nearly all but mixes in distinct images | [retrieval_baseline.md](../research/retrieval_baseline.md), `src/services/curation/item_scores/near_dup.py` |

### 1.2 Decisions per arm (from the outcome note, section 3)

| Arm | Decision | Key number |
|---|---|---|
| IVF | stays | best of the arms on `rare` items |
| Average linkage inside the detector class | parked | A4 BCubed recall 0.288 against bar 0.329; cut moves between sets (0.346, 0.345, 0.515) |
| Whitening, k-reciprocal, DBA | killed | precision at matched recall -0.033 / -0.065, -0.014, -0.011 |
| Novelty flag by distance | killed | AUROC 0.838 and 0.851 against 0.90 |
| Judge-gated merge (Z3) | killed | +2.2 % effort against a 10 % saving bar |
| Split by descent | killed | 71 % and 9 % of impure clusters fixed; bar 80 % |
| Within-class roots | killed | 0.526 decisions per item against limit 0.417 |
| Detector-class roots, with or without guard | **parked as the lead** | Z4: 0.293 (key) and 0.377 (key + guard) decisions per item against IVF 0.834; precision 0.982 and 0.988; open-set bar unreadable (2 of 20 seeds reached 30 propagated) |
| Leiden CPM + CHAMELEON filter | parked | 539 real COCO crops: 0.944 / 0.195 against IVF 0.905 / 0.088, filter tuned in-sample; synthetic recall 0.79 to 0.86 does not carry over |
| Consensus Leiden, HDBSCAN leaf, FINCH | killed | 5x time for equal quality; HDBSCAN about 80 % noise; FINCH has no granularity control |
| cuGraph Leiden (modularity) | not usable as Leiden CPM | over-merges, false-merge 0.236 at 20k to 0.609 at 500k (synthetic) |
| cuGraph ECG | parked | 0.953 / 0.899 at 20k synthetic only |
| Encoder swap | not adopted | WB2: no challenger passes the registered gain; PE-Core-G best everywhere but about 8x encode time and 5x memory |
| Letterbox for whole images | not adopted | no gain reaches 0.03 on CARS196 |
| Haiku judge | research tooling only | COCO crops: pair kappa 0.821, flag precision 0.827 (pass); Open Images vehicles 0.611, 0.454 (fail) |
| Local VLM judge (4B) | killed as an authority | pair accuracy 0.51 on COCO crops |

### 1.3 Lessons that shape this plan

1. **Most fragmentation is configuration.** IVF K = n // 6 gives 6 to 8 crops per cluster and
   forces pair recall near zero; a global average-linkage AHC on raw vectors reached 0.92 / 0.20 on
   COCO crops with nothing unassigned ([consolidated note](../research/clustering_research_consolidated.md) section 1).
2. **Above pair recall about 0.3, neighbourhoods are the limit.** P@10 is 0.82 overall, 0.78 for
   crops under 32 px and 0.74 for crops longer than 2:1. Chaining is a property of the space.
3. **The crop preprocessing defect is untested.** PE crops are resized then centre-cropped; the
   median crop keeps 69 % of its long side and 21 % of crops are longer than 2:1. Letterbox was
   tested on whole images (CARS196), never on curation crops. B5c ("crops at model size") touches
   the same path.
4. **The space is anisotropic.** Mean random-pair cosine 0.50 on COCO; effective rank 104 to 122 of
   1,024. A fixed cosine threshold means different things per project, so every threshold is
   calibrated per project and space.
5. **The label effort is in the gate, not the method.** One gate per detector class saves about
   2.1 to 2.4x the decisions of IVF; the open risk is minority exposure of open-set items.
6. **Benchmarks at the ceiling decide nothing.** CARS196 Recall@1 above 0.95 for four encoders made
   WB2 exploratory. The product task (crop clustering and the loop's effort on COCO and Open Images)
   must be the primary encoder test.
7. **A held-out half wears out.** The 3,000-item held-out half was read by three waves; acceptance
   needs a fresh set once a half has been read repeatedly.
8. **The judge needs balanced sheets.** The layout calibration was vacuous because the judge said
   "different" to every sheet; calibration sheets must mix clean and contaminated clusters.

## 2. Research questions

| RQ | Question | Why it matters for curation | Issue | Priority |
|---|---|---|---|---|
| RQ-1 | Do detector-class roots with the hold-back guard keep open-set items safe (open-set propagated precision >= 0.90) on a set with at least 1,200 open-set crops? | The one lead that cuts labelling decisions to about a third of IVF's. Wrong labels on unknown animals become training noise | #398 (Z5, then Z5-I) | 1 (ship track) |
| RQ-2 | Which near-duplicate thresholds catch real re-encodes and resizes without grouping distinct photos, per path (core, frame, crop)? | Duplicate groups are must-links, one vote per group in gates, and the train/test split guard. Today the core default misses most recompressions | #416 | 1 (ship track) |
| RQ-3 | Does a crop preprocessing change (letterbox, context margin) or another encoder improve clustering and the loop's effort on real crops, enough to pay for a re-embed? | Embedding quality bounds every method above recall 0.3; a swap costs a full re-index (#365) | #347, #365, new N-3 | 2 |
| RQ-4 | Does a global graph method (average-linkage graph HAC or Leiden CPM with a filter) beat IVF on a fresh held-out real set at 3k to 860k items, and does a GPU path agree with it? | Higher recall at the same precision means fewer, larger clusters to review | #346, new N-4 | 2 |
| RQ-5 | Can a partition keep stable ids and human corrections across re-runs and additions (protected moves 0, id stability by membership overlap)? | Without it, every re-run destroys review work; it is the precondition for shipping any RQ-4 winner | #346 (former #343, #344), new N-5 | 3 (only if RQ-4 wins) |
| RQ-6 | Do attribute signals (near-duplicate group, detector class, view tag, colour, EXIF time or burst) as must-link or veto evidence raise precision or recall at a fixed false-merge rate? | Some links embeddings cannot make (front and rear of one object); some merges they wrongly make | #400 | 4 (parked) |
| RQ-7 | Can a plate identity key, with privacy modes, give links with precision >= 0.99 at useful coverage? | The strongest same-vehicle link, but a privacy risk | #401 | 5 (parked, owner-run) |
| RQ-8 | For which kinds is the visual judge trustworthy, at which layout, and how well does it agree with a blind human? | Every quality claim above rests on visual checks; an unchecked judge makes them weightless | #406 | 1 (enables the rest) |
| RQ-9 | Which face grouping method beats union-find at 0.7 on chaining and purity, how should new faces be assigned with abstention, and how does it scale? | Single linkage chains look-alikes; the route runs O(n^2) inside the request | #346 (faces), new N-1, N-6 | 2 |
| RQ-10 | Is the curation frame vector (PE) as good as the core whole-image vector for whole-image search and grouping? | Decides whether `target=frame` (#435) and `group_by=image` (#436) belong in curation search | #435, #436 | 4 |

Fine-grained make and model grouping stays inside RQ-3: it was shown to be an embedding problem,
and an adapter is considered only if RQ-3 and RQ-4 both stall (stop rule in section 7).

## 3. Methods to compare, with the library and the documented pattern

### 3.1 Libraries and their documented limits

| Library (licence) | Used for | Documented fact this plan relies on |
|---|---|---|
| [FAISS](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU) (MIT) | exact and IVF kNN, range search (CPU), GPU flat and IVF kNN; CAGRA through the cuVS backend since 1.10 ([GPU Faiss with cuVS](https://github.com/facebookresearch/faiss/wiki/GPU-Faiss-with-cuVS)) | GPU indexes cap `k` and `nprobe` at 2048 and large adds and searches must be batched; GPU indexes must be converted to CPU to save |
| [cuVS](https://github.com/rapidsai/cuvs) (Apache-2.0) | GPU kNN graph build (CAGRA, brute force, NN-descent) | graph build and ANN search on GPU; no Leiden |
| [cuGraph](https://docs.nvidia.com/cugraph/latest/api_docs/api/cugraph/cugraph.leiden/) (Apache-2.0) | GPU ECG, connected components; Leiden only as a contrast arm | `cugraph.leiden(G, max_iter, resolution, random_state, theta)` optimises modularity only and takes no initial partition |
| [leidenalg](https://leidenalg.readthedocs.io/en/stable/reference.html) (GPL-3.0) with python-igraph | Leiden CPM | `CPMVertexPartition(initial_membership=..., resolution_parameter=...)`; `optimise_partition(..., is_membership_fixed=...)` pins locked nodes; CPM is well defined for negative edge weights (cannot-link as negative edges) |
| [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html) (BSD-3) | `AgglomerativeClustering` (average or complete linkage, `connectivity`, `distance_threshold`); [`HDBSCAN`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) (contrast only; labels noise -1); `homogeneity_completeness_v_measure`, `adjusted_mutual_info_score`; `IsotonicRegression` for calibration | HDBSCAN was killed on real data (about 80 % noise); not re-opened unless a new embedding changes the density picture |
| [OpenSearch k-NN](https://docs.opensearch.org/latest/vector-search/specialized-operations/radial-search-knn/) | incremental assignment (radial search with `min_score`), nested box kNN | the incremental face and item assignment reads candidates from the index, never a full matrix |
| [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html) and [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) | Wilson intervals, bootstrap intervals | judge purity reported with Wilson 95 % intervals; metric deltas with a 1,000-resample bootstrap |

Licence note: leidenalg is GPL-3.0. GPLv3 section 13 allows combining with AGPLv3 code, but a
product dependency needs an owner decision and an ATTRIBUTION row (open question OQ-4). Until then
leidenalg stays a research dependency (`scripts/research/fine_grained_requirements.txt`), and the
product fallback is scikit-learn average linkage on a sparse connectivity graph.

Existing harness (reuse, do not rewrite): `scripts/bench/cluster_eval.py`,
`scripts/bench/cluster_eval_metrics.py` (pairwise, BCubed, AMI, risk-coverage, attribute pair
scores, moved protected items), `scripts/bench/scorecard.py` (envelope, synthetic refusal),
`scripts/research/cluster_diag/` (vectors, loop simulation, Z4 bars, chaining), the judge scripts
(`scripts/research/visual_judge.py`, `judge_*.py`, `src/services/labeling/contact_sheet.py`),
`scripts/bench/retrieval_eval*.py`, `scripts/research/leiden_refine_proto.py` and
`gpu_methods.py`. New metric code: V-measure through scikit-learn and a purity helper, each with a
hand-computed unit test that fails on a wrong formula.

### 3.2 Arms per research question

| RQ | Baseline | Arms | Library | Documented pattern / prior evidence |
|---|---|---|---|---|
| RQ-1 | IVF with the loop's gate | `key` (one root per detector class name plus the loop's splitter), `key_guard` (plus the label-free hold-back at the 95th percentile distance) | loop simulator in `scripts/research/cluster_diag/` | Z4 registration ([cluster_config_decision.md](../research/cluster_config_decision.md) 0A.2); roots keyed by class name |
| RQ-2 | core 0.99, frame 0.98, crop 0.98 | thresholds 0.90 to 0.995 in 0.005 steps; single linkage (today) against complete linkage inside a candidate group (stops chaining at lower thresholds); per-path defaults | FAISS range search (exact, CPU); scikit-learn complete linkage on the candidate groups | union-find chaining is the known false-merge cause ([cross_domain_clustering_lessons.md](../research/cross_domain_clustering_lessons.md)) |
| RQ-3a | PE-Core-L, resize + centre crop (the B5c crop path) | letterbox; context margin 10 % and 20 %; letterbox + margin | the product encoder through a throwaway project; `src/utils/pe_preprocess.py` variants behind a research flag | consolidated note E2 (never run); WB2 bar B2 shape |
| RQ-3b | PE-Core-L | PE-Core-G, SigLIP 2, MobileCLIP2-S2, the #347 candidate encoder, one LLM embedder as reference only | open_clip, timm, transformers or vLLM (per #347 owner comments) | [embedding_model_eval_plan.md](../research/embedding_model_eval_plan.md); WB2 bars B1 and B4 (a production encoder needs a text tower) |
| RQ-4 | IVF | global average-linkage graph HAC on a mutual-kNN graph (k 30) with a dev-calibrated cut; Leiden CPM (gamma log-grid 0.0003 to 0.01) with and without the CHAMELEON-style filter; GPU: cuVS or FAISS GPU kNN graph feeding the same CPU methods, and cuGraph ECG | FAISS, cuVS, scikit-learn, leidenalg, cuGraph | Traag et al. 2019 ([Leiden](https://www.nature.com/articles/s41598-019-41695-z)); prototype findings ([leiden_refinement_prototype.md](../research/leiden_refinement_prototype.md)) |
| RQ-5 | re-fit from scratch, ids renumbered | ids kept by membership Jaccard >= 0.5 against the previous partition; locked nodes pinned (`is_membership_fixed`); must-links by graph contraction (validated groups and near-duplicate groups become one node); cannot-links as negative CPM edges or a hard check in the merge validator; incremental assignment with accept, review and new-microcluster bands | leidenalg, scikit-learn, FAISS | cross-domain lessons 6.2 to 6.4; Immich's deferred pass for faces ([facial recognition docs](https://docs.immich.app/features/facial-recognition)) |
| RQ-6 | embedding only | near-duplicate group as must-link; detector class as a veto across coarse classes when both are confident; zero-shot view tag from the PE text tower against stored vectors (no pixels); dominant colour veto at identity level only; EXIF burst as must-link only with high embedding similarity | RQ-5 machinery; Pillow for colour; the existing EXIF reader if any, else Pillow `getexif` | rule from the parked plan 6.7: no attribute decides alone |
| RQ-7 | none | plate key tiers `confident`, `partial`, `none`; must-link on equal confident keys plus an embedding floor; veto on edit distance >= 2 | existing region text fields and PP-OCRv5; HMAC-SHA256 from the Python standard library for `hash_only` | parked plan 6.8 design (privacy modes, purge with secret rotation) |
| RQ-8 | the E0 judge result | balanced layout calibration (half clean, half contaminated sheets); Haiku against the local VLM; grid 3x3, 4x4, 6x4 at 128 and 192 px; Message Batches for cost | the existing judge scripts and the `anthropic` SDK in the research venv | position-bias controls and planted golds ([MLLM-as-a-Judge](https://arxiv.org/abs/2402.04788), [Zheng et al. 2023](https://arxiv.org/abs/2306.05685)) |
| RQ-9 | union-find at 0.7 (production) | Immich-style core-point rule (cosine distance 0.5, min 3 faces, detector score 0.7); average and complete graph HAC on mutual kNN; Leiden CPM; each with and without the same-image cannot-link; incremental assignment to the nearest of up to 8 prototypes per person with accept / review / unassigned bands calibrated by isotonic regression; approximate range search (FAISS IVF or HNSW) against exact | FAISS, scikit-learn, leidenalg, OpenSearch radial search | Immich [config docs](https://docs.immich.app/install/config-file/); [approximate rank-order](https://arxiv.org/abs/1604.00989); learned GCN clusterers out of scope (their training data is withdrawn) |
| RQ-10 | core MobileCLIP whole-image vector | curation PE frame vector | `scripts/bench/retrieval_eval.py` | [retrieval_baseline.md](../research/retrieval_baseline.md) "the experiment that separates them is whole-image PE embeddings on the same sample (not done)" |

Why the hold-back guard, contraction and pinned nodes are not hand-rolled clustering: they are
pre- and post-processing around library calls (`optimise_partition` with fixed membership, graph
contraction before `AgglomerativeClustering`), which is the documented way to impose constraints
when no maintained constrained-clustering library fits. No maintained Python library implements
must-link and cannot-link constrained graph clustering at this scale (checked: scikit-learn has
none; leidenalg offers fixed membership and signed CPM, which this plan uses).

## 4. Evaluation protocol

### 4.1 Datasets

| Tier | Set | Size | Ground truth | Used by |
|---|---|---|---|---|
| Published | existing pin `cluster_eval_3000.json` | 944 images, 3,961 crops, 2,950 labelled, 86 classes | COCO and Open Images boxes matched to crops at IoU >= 0.5 | regression row for every RQ (its held-out half is worn: report, never accept on it) |
| Published | existing pin `cluster_eval_fresh.json` and the Z2/Z4 Open Images sets | 33,580 crops (Z2), 7,209 labelled held-out crops (Z4) | Open Images boxes | RQ-1, RQ-3, RQ-4 |
| Published | **new** Open Images V7 train-split open-set set for Z5 | about 8,000 to 20,000 images until at least 1,200 held-out open-set crops (classes outside the detector's label space) exist | Open Images boxes, open-set = class names absent from the detector's names | RQ-1 |
| Published | **new** fresh acceptance set: COCO val2017 (5,000 images, about 36,800 instances) plus an Open Images V7 validation sample (41,620 images in the split), disjoint from every earlier pin | about 10,000 images | human boxes | RQ-3, RQ-4 acceptance |
| Published | COCO train2017 | 118,287 images, about 860,000 instances | human boxes | RQ-4 scale with quality (replaces every synthetic 1M claim) |
| Published | `coco_bench_2000.json` (retrieval pin) | 2,000 images, 5 captions each | COCO captions and categories | RQ-2 originals, RQ-3b text-to-crop, RQ-10 |
| Published | near-duplicate set built from 5,000 distinct COCO val and Open Images originals | 5,000 originals x 8 transforms (JPEG q60 and q30, half and quarter size, 90 % and 80 % centre crop, brightness x1.15, small rotation) | positives by construction; negatives = distinct originals; plus a person-checked sample of 200 natural near-pairs found above 0.95 | RQ-2 |
| Published | Open Images V7 "Human face" boxes (existing 1,500-image pin, plus up to 100,000 train images for scale) | as stated | no identities: chaining and false merges judged by a person on samples (the local VLM ranks only) | RQ-9 scale and chaining |
| Published | `oi_plates_300.json` | 300 images | a person transcribes locally; text never committed | RQ-7 read quality |
| Local only | LFW, CALFW, CPLFW | 13,233 / 12,174 / 11,652 images | author identity labels | RQ-9 identity quality; aggregates published only if the owner allows (OQ-1) |
| Local only | CARS196, CUB-200-2011, SOP | 16,185 / 11,788 / 120,053 images | author labels | RQ-3b secondary rows (CARS196 is at the ceiling; CUB and SOP carry the signal) |
| Owner-run | private photo set | owner's choice of sample | validated items, a frozen holdout | optional repeat of the RQ-1, RQ-3, RQ-4, RQ-6, RQ-7, RQ-9 winners; numbers only |

Every new set gets a pin manifest under `scripts/datasets/manifests/` (counts, licence, a
`manifest_sha256`), a fetcher flag where needed (the Open Images fetcher needs train-split support
for Z5, as #398 notes), and a row in the #346 "real-world dataset testing" checklist.

### 4.2 Splits

- Split by image (never by crop), seeded, 50 % dev and 50 % test. Calibration (cuts, gamma,
  thresholds, bands) only on dev. The test half is read once per method family.
- Open-set splits are class-disjoint.
- A set whose test half has been read by two method families is retired for acceptance and kept
  as a regression row.
- Near-duplicates never straddle dev and test (frame cosine >= 0.98 groups are split as a unit).

### 4.3 Metrics

| Group | Metrics | Notes |
|---|---|---|
| Partition quality | purity (majority class share) and inverse purity; homogeneity, completeness, V-measure ([Rosenberg and Hirschberg 2007](https://aclanthology.org/D07-1043/)); BCubed precision, recall, F (headline, [Amigo et al. 2009](https://doi.org/10.1007/s10791-008-9066-8)); pairwise precision, recall, F; AMI; false-merge and false-split rates | strata: low resolution, elongated (> 2:1), rare, similar classes, person, open-set |
| Abstention | unassigned fraction, risk-coverage curve | BCubed counts an abstained item as a singleton (harness rule) |
| Stability | permutation ARI on identical vectors (must be 1.000), seed ARI, id stability (share of clusters keeping their id), protected items moved (must be 0) | |
| Human effort to label | U1 decisions per item and U2 propagated precision from the loop simulator (oracle labeler, and a 10 % noisy labeler beside it); open-set propagated precision; labels per correctly labelled item | the loop's gate: theta 0.9 homogeneity from a sample |
| Faces | persons per identity (fragmentation), identities per person (contamination), chaining incidents, incremental accept precision at coverage, corrections kept across a re-run | identity labels only; class metrics never use identity labels |
| Near-duplicates | recall per transform, false-group rate on distinct pairs, group purity | per path |
| Retrieval | recall@1/5/10, MRR@10, nDCG@10 with bootstrap intervals | the retrieval harness |
| Cost | wall time, peak RSS and VRAM, host load at start and end | a run starting at 1-minute load above 16 is noisy and repeated |

### 4.4 Visual judge procedure

1. **Agreement first, per kind.** Before a judge's verdict counts on a kind, it is scored against
   ground truth on a labelled public sample of that kind: pair kappa >= 0.7 and flag precision
   >= 0.8 (the bars of [visual_judge_e0.md](../research/visual_judge_e0.md)). Below them the judge
   only ranks clusters for a person.
2. **Sheets.** The existing builder (`src/services/labeling/contact_sheet.py`): seeded member
   selection (prototypes, near members, 2 to 3 far members, random fill), letterboxed cells, cell
   order shuffled, manifest with hashes. The layout is the one RQ-8 picks.
3. **Sample.** Per experiment at most 20 cluster sheets, 10 pair sheets and 20 search sheets for
   Haiku (owner decision O3: small subsets). Half targeted (borderline merges, largest clusters,
   method disagreements), half a seeded uniform control; targeted sheets are never pooled into
   the purity estimate.
4. **Controls.** About 10 % planted known-same and known-different sheets from labelled public
   data; a batch whose controls fail is discarded. 10 % of sheets re-rendered in a second order to
   measure position bias.
5. **Privacy.** Face and plate sheets go to the local VLM or a person only; the sender refuses a
   hosted call for those kinds (already tested in `tests/test_contact_sheet.py`).
6. **Human re-judge.** A blind re-judge of a seeded 5 % of verdicts (at least 20 per run) by the
   owner or a delegate; judge-vs-human kappa tracked per kind; a drop below 0.7 suspends the judge
   for that kind.
7. **Reporting.** Judge purity and false-merge estimates with Wilson 95 % intervals and a
   cluster-level bootstrap, beside the ground-truth metrics, never instead of them. A run that
   cannot reach a half-width of 0.10 within budget reports "insufficient sample", not a pass.

### 4.5 Spread and repetition

- Three seeds and one row permutation per headline row; 20 seeds for loop simulations (as Z4).
- A gain counts only if it is larger than twice the run-to-run spread and its bootstrap interval
  excludes 0 (the program's B5 rule).
- Every run records git sha, dataset manifest hash, model and engine versions, embedding field,
  seed, GPU id, `nvidia-smi` snapshot, host load, wall time, peak RSS and VRAM.
- Raw output goes to `artifacts_local/` (gitignored). The small JSON files that back a published
  number go to `docs/benchmarks/<experiment>/` with a `SHA256SUMS`, and are never pruned.

### 4.6 Pass bars that make a method shippable

Registered in phase R0 (a docs commit) before any score. Verdicts: PASS, FAIL, INCONCLUSIVE.

| RQ | Ship bar (all must pass, on the test half of a fresh set) |
|---|---|
| RQ-1 | Z4 bars unchanged: U2 >= 0.95; U1 <= 0.5 x IVF's U1; open-set U2 >= 0.90 with at least 30 open-set items propagated per run on average. Adoption: the cheaper of `key` and `key_guard` that passes |
| RQ-2 | per path: false-group rate on distinct pairs <= 0.5 % (at most 1 in 200 groups mixes distinct photos in the person-checked sample); recall >= 0.90 on q60 and half-size; >= 0.80 on 90 % crop; no worse than today on every other transform |
| RQ-3a | a preprocessing variant: BCubed F at the calibrated cut +0.03 or more with the interval above 0, elongated-crop P@10 gain >= 0.03, no stratum worse by more than 0.01, U1 not worse |
| RQ-3b | an encoder: the RQ-3a bar on COCO and Open Images crops, plus WB2 B1 on CUB and SOP (no loss beyond 0.01), plus a text tower (WB2 B4), plus encode time within 2x of PE-Core-L on the B5 pipeline |
| RQ-4 | corrected acceptance A1 to A7 of [cluster_config_decision.md](../research/cluster_config_decision.md) section 6 (BCubed precision >= 0.90, non-person >= 0.90, low-res >= 0.85, BCubed recall >= 2x IVF, pair recall >= 3x IVF, no stratum with >= 100 crops worse than IVF by more than 0.02, permutation ARI 1.000), on the fresh set and on COCO train2017; U1 not worse than IVF; at 860k items within 2 h wall on CPU or the agreed GPU path |
| RQ-4 GPU | agreement gate: on COCO train2017, purity within 0.01 and label agreement (ARI between GPU and CPU partitions) >= 0.95, plus the same A-bars |
| RQ-5 | protected moves 0; id stability >= 0.95 of clusters across a re-run with 10 % new items; 100 % of human merges, splits and names kept; permutation ARI 1.000 |
| RQ-6 | a signal: at a fixed false-merge rate, BCubed recall +0.02 or more (must-link) or false merges blocked >= 3x true merges lost (veto), on a stratum with at least 100 affected pairs |
| RQ-7 | confident-key precision >= 0.99; plate-link precision >= 0.99; coverage reported; no plate text in logs, exports or committed files (tests); purge leaves 0 fields and keys |
| RQ-8 | per kind: pair kappa >= 0.7 and flag precision >= 0.8 against ground truth; human-vs-judge kappa >= 0.7 on the 5 % re-judge; a layout chosen by a non-vacuous calibration |
| RQ-9 | BCubed precision >= 0.97 on LFW plus CPLFW identities with at least 2 images; chaining incidents fewer than union-find at matched recall; incremental accept precision >= 0.99; corrections kept 100 %; memory O(n k) |
| RQ-10 | PE frame recall@10 within 0.02 of MobileCLIP on the 2,000 pin, or better; otherwise #435 closes as "not in curation" |

## 5. Experiments

Compute figures are estimates for an A6000 on a quiet host; each experiment records the real
cost. GPU work runs in a research window (RW) on the GPU agreed with the peer sessions (GPU 2 or
GPU 0; GPU 1 never, unless its owner frees it). CPU work runs in a CPU slot: any period with no
sweep, live run or GPU window of the program, capped at 8 threads and `nice`, after checking
`uptime` and asking the peer sessions. Research windows never overlap a program window and start
after the v0.7.0 tag.

| E | RQ | What | Data | Compute | Window | Output |
|---|---|---|---|---|---|---|
| E-0 | all | Phase R0: register bars, extend the fetchers (Open Images train split; the near-duplicate transform builder), add V-measure and purity helpers with red/green unit tests, add the COCO-to-Open-Images name map | none | about 1 CPU h | CPU slot, docs and research scripts only | bar registrations committed; scripts tested |
| E-1 | RQ-1 | Z5: fetch the open-set-rich Open Images train set; ingest into a throwaway project; export vectors; simulate `key`, `key_guard`, `within_key`, IVF over 20 seeds | new Z5 set | GPU about 1 h (ingest); CPU about 3 h | RW-1 + CPU slot | `docs/research/cluster_config_z5.md`; #398 verdict |
| E-2 | RQ-2 | encode 5,000 originals and 40,000 variants on all three paths (core, frame, crop); sweep thresholds and linkage; person checks 200 natural near-pairs | near-duplicate set | GPU about 1 h; CPU about 1 h; person about 1 h | RW-1 + CPU slot | `docs/research/near_duplicate_threshold.md`; recommended defaults per path |
| E-3 | RQ-3a | re-encode crops of the 3,000-item set, the fresh set and the Z2 set under four preprocessing variants on the B5c crop path; score P@10 by stratum, BCubed at the calibrated cut, loop U1/U2 | published sets | GPU about 2 h; CPU about 2 h | RW-1 + CPU slot | `docs/research/crop_preprocessing_eval.md` |
| E-4 | RQ-3b | encode the same crops (and CUB and SOP locally) with each encoder arm at the RQ-3a winner; score as E-3 plus text-to-crop retrieval on the 2,000 pin; one-day time box per arm (owner decision on #347) | published sets plus local-only rows | GPU about 8 to 10 h | RW-2 | `docs/research/encoder_eval_curation.md`; #347 and #365 verdicts |
| E-5 | RQ-4 | build mutual-kNN graphs (k 30) for the fresh set and COCO train2017; calibrate cuts and gamma on dev; read the test half once: IVF, graph HAC, Leiden CPM with and without the filter | fresh set, COCO train2017 | GPU about 1 h (kNN graphs); CPU about 12 to 16 h (HAC and Leiden at 860k) | RW-2 + CPU slot | `docs/research/global_refinement_eval.md` |
| E-6 | RQ-4 | GPU arms on the same graphs: FAISS GPU or cuVS kNN feeding the CPU methods; cuGraph ECG; agreement gate against the CPU partition | COCO train2017 | GPU about 3 h | RW-2 | section of the E-5 note; agreement verdict |
| E-7 | RQ-5 | on the E-5 winner only: permutation, re-run with 10 % new items, locked and validated items pinned, must-link contraction, cannot-link edges; incremental assignment with bands | fresh set | CPU about 3 h | CPU slot | `docs/research/partition_stability_eval.md` |
| E-8 | RQ-6 | feasibility first: share of public images with usable EXIF time; then near-duplicate must-link, class veto, view tag, colour on the RQ-5 machinery | fresh set, Z2 vehicles | CPU about 4 h (view tag uses the PE text tower on CPU) | CPU slot | `docs/research/attribute_signals_eval.md` |
| E-9 | RQ-7 | only after the privacy design review: read quality and coverage on the 300-plate pin and the Open Images vehicle sets; linking is owner-run on the private set | plate pin, vehicle sets | GPU about 1 h (segmenter and OCR); person about 1.5 h (transcription, local) | RW-3 | `docs/research/identity_text_eval.md` (aggregates only) |
| E-10 | RQ-8 | balanced layout calibration; Haiku against the local VLM per kind (crops, vehicles, whole images); 5 % human re-judge; the person-checked make/model label sample (FG-0) | COCO, Open Images, Commons set C | Haiku at most about 300 sheets (about 0.3M input tokens, Message Batches); local VLM GPU about 2 h; person about 3 h | RW-3 (local VLM) + API | update of `docs/research/visual_judge_e0.md` or a successor; #406 closed |
| E-11 | RQ-9 | embed LFW, CALFW, CPLFW and up to 100,000 Open Images face images through SCRFD and ArcFace; arms of 3.2; calibrate bands on dev; incremental assignment on a 20 % holdout; approximate against exact range search at 100k real faces | face sets | GPU about 3 h (detection and embedding); CPU about 4 h; person about 1.5 h (chaining check on Open Images samples) | RW-3 + CPU slot | `docs/research/face_grouping_eval.md` |
| E-12 | RQ-10 | encode whole-frame PE vectors for the 2,000 pin; text-to-image and image-to-image against the core path; whole-image grouping by label-set Jaccard purity | retrieval pin | GPU about 0.5 h; CPU about 0.5 h | RW-1 | section in `docs/research/retrieval_baseline.md` (append, never rewrite) |

### 5.1 Batching

| Window | Contents | Estimated GPU time | Why batched |
|---|---|---|---|
| RW-1 | E-1 ingest, E-2 encodes, E-3 re-encodes, E-12 frame encodes | about 4 to 5 h | one stack (`-p op-rs-rw1`), one model set, encode once and store vectors |
| RW-2 | E-4 encoder arms, E-5 kNN graphs, E-6 GPU arms | about 12 to 14 h | encoder arms need E-3's winner; graph arms need the vectors |
| RW-3 | E-9, E-10 local VLM, E-11 faces | about 6 h | VLM and face models on one card; E-9 waits for the privacy review |
| CPU slots | E-0, scoring of every E, E-1 simulation, E-5 HAC and Leiden, E-7, E-8 | about 35 to 45 CPU h | no GPU; run between program sweeps |

Totals: about 22 to 25 GPU hours over three windows, about 40 CPU hours, about 8 hours of
owner or delegate time (human re-judge, transcription, checks), and a few hundred Haiku sheets.

### 5.2 Order

1. R0 (E-0): docs and scripts, any time.
2. Ship track after the tag: RW-1 (E-1, E-2, E-3, E-12) and E-10's Haiku half.
3. RW-2 (E-4, E-5, E-6), using E-3's preprocessing winner.
4. RW-3 (E-10 local half, E-11; E-9 only if the owner approves the privacy design).
5. E-7 only if E-5 or E-6 passes; E-8 only if E-7's machinery exists.

## 6. How winners land in the product

Nothing lands without a PASS, a visual check, a human G3 review in Cropwright and the owner's yes.
Each landing is one feature PR on a branch, with red/green tests, `make contracts` in the same
commit as any wire change, docs and a CHANGELOG entry.

### 6.1 Job shape (after B2)

| Work | Celery task (proposed) | Queue | Lease (B1) | Notes |
|---|---|---|---|---|
| Residual clustering with class roots (RQ-1) | the existing auto-label task and `op.periodic.cluster_refresh` (B2 2D) | `autolabel` | `autolabel` per project | no new task: the partition reader of the loop changes |
| Near-duplicate grouping (RQ-2) | the existing item-scores task (`op.jobs.item_scores`) | `jobs` | `item_scores` | the threshold default changes; a re-score writes new `dup_*` fields through the guarded writers |
| Global refinement run (RQ-4) | `op.jobs.cluster_refine` | `autolabel` (GPU-capable worker) | `cluster_refine` per (project, space, kind) | writes a candidate partition, never the active one; the GPU path only after its agreement gate passed for that backend version |
| Partition activation (RQ-5) | none: a request that does an OCC write of the active-partition pointer on the project settings document | | | refuses a stale revision; rollback is the same write to the previous partition |
| Incremental assignment (RQ-5, RQ-9) | inside ingest for items (as `ivf_ingest` does today); `op.jobs.persons_assign` for faces | `jobs` | none (idempotent per face) | abstains into the review band |
| Person grouping (RQ-9) | `op.jobs.persons_cluster` | `jobs` | `persons_cluster` (global scope; faces are core-only per O1) | today `POST /persons/cluster` runs O(n^2) inside the request; see N-1 |

Every task follows the B2 `JobTask` wrapper: bind the project, take the lease, heartbeat, run,
fence the terminal write, release. No in-process locks; caches stay caches.

### 6.2 Stored partition and versioning

- A partition record per (project, embedding space, kind, level): `partition_id`, parent
  `partition_id`, method, parameters, embedding model and version, calibration id (thresholds are
  valid only for their space and model), dataset counts, metrics summary, status (`candidate`,
  `active`, `retired`), created time.
- Membership of a candidate lives in the project's state directory as a JSONL artifact (as the
  loop's job directory does today); only the active partition is written onto items, through the
  guarded writers (`cluster_write_guard`) and the lock rule (`src/clients/occ_locks.py`).
- Stable ids by membership overlap (Jaccard >= 0.5); human merges, splits and names are
  anchors; machine merges and propagations keep their own provenance and are never validated.
- Where the record lives (a new project-owned index role or the settings document) is a design
  decision for the N-5 design doc; either way the OpenSearch guard must cover it and the project
  delete must remove it.

### 6.3 API and wire

Proposed, not built (no leading slash, as the docs check requires for routes that do not exist):

- `curation/projects/{project}/clusters/partitions` (list, with metrics) and
  `curation/projects/{project}/clusters/partitions/{partition_id}/activate`;
- `curation/projects/{project}/clusters/suggestions` (merge and split proposals with evidence) and
  accept or reject actions that write must-link or cannot-link records;
- item and box wire fields: `partition_id`, `cluster_level`, `cluster_state`
  (`assigned`, `review`, `unassigned`), prototype ids on cluster cards;
- faces: a job-id response for person grouping (202), and merge and split suggestions for persons.

All of these go through the one serializer (`src/services/curation/wire.py`), regenerate
`contracts/`, and update [curation_api_contract.md](curation_api_contract.md).

### 6.4 Cropwright review flow

The backend serves data and decisions; the frontend renders them.

- Cluster page: partition version and its metrics, prototypes per cluster, the review band
  (abstained items), merge and split suggestions with evidence and accept or reject.
- People view (faces): persons with prototypes, merge suggestions, split proposals, rename, purge
  dry run.
- Activation screen: candidate against active partition, a diff summary, activate and roll back.
- Every screen is checked in a browser with full-page screenshots at desktop and 430 px width.

### 6.5 Issue map

| Landing | Issue | Action |
|---|---|---|
| Class roots + guard in the loop (Z5-I) | #398 | implement only on E-1 PASS; else close with the evidence |
| Near-duplicate defaults per path | #416 | close with the E-2 note and the default change |
| Crop preprocessing or encoder change | #347, #365 | #347 closes with the E-4 verdict; #365 unblocks only if a candidate passes |
| Judge calibration | #406 | close with E-10 |
| Attribute signals | #400 | stays parked unless E-8 passes |
| Plate identity | #401 | stays parked; E-9 needs the privacy design first |
| Frame target, group by image | #435, #436 | #435 decided by E-12; #436 only if a frame-level view is wanted |
| Tracker | #346 | one results row per experiment |
| **N-1** (new) | `feat(faces): run person grouping as a Celery job` | `POST /persons/cluster` computes an O(n^2) grouping inside the request (about 4 min at 200k faces). Move it to `op.jobs.persons_cluster` with a lease and a 202 + job id. Independent of the research outcome; candidate for a post-B2 batch |
| **N-2** (new) | `research(clustering): register bars and extend fetchers for the research waves` | E-0 |
| **N-3** (new) | `research(embed): crop preprocessing (letterbox, context margin) on COCO and Open Images crops` | E-3 |
| **N-4** (new) | `research(clustering): global refinement on a fresh held-out real set, 3k to 860k, with an agreement-gated GPU path` | E-5, E-6 |
| **N-5** (new, only on E-5/E-6 PASS) | `feat(clustering): versioned partitions with stable ids, activation and merge/split suggestions` | design doc first (6.2, 6.3), then E-7 |
| **N-6** (new) | `research(faces): identity grouping, calibrated incremental assignment and scale on public faces` | E-11 |

### 6.6 Found while reading (for the coordinator, not research)

- `src/services/curation/item_scores/near_dup.py` `_threshold()` catches `ValueError` from a bad
  `OP_CROP_DUP_THRESHOLD` and silently uses the default. That is fail-open on a misconfiguration;
  the house rule is fail closed. Small fix, independent of #416.
- `POST /persons/cluster` is a heavy job in a request handler and is not in the B2 inventory of 17
  job families (N-1).

## 7. Stop rules, risks and open questions

### 7.1 Stop rule per line

| RQ | Stop when | Then |
|---|---|---|
| RQ-1 | E-1 FAIL, or INCONCLUSIVE again with at least 1,200 open-set crops | IVF stays; close #398 with the evidence |
| RQ-2 | no threshold meets both bars on a path | keep today's default on that path, document the trade-off, offer it as a project setting only if the owner wants one |
| RQ-3a | no variant passes | keep resize + centre crop; record the elongated-crop deficit as a known limit |
| RQ-3b | no encoder passes, or the passing one lacks a text tower or costs more than 2x | PE-Core-L stays; #365 stays blocked; the adapter question is not opened |
| RQ-4 | E-5 FAIL on the fresh set | park graph refinement for good in #346; E-6 and E-7 are not run |
| RQ-4 GPU | agreement gate fails | GPU arms are not adopted; CPU result stands |
| RQ-5 | E-7 cannot reach protected moves 0 and id stability 0.95 | the RQ-4 winner does not ship (a partition that destroys review work is worse than IVF) |
| RQ-6 | feasibility shows too little public signal, or no signal passes | #400 stays parked; owner may run it on the private set |
| RQ-7 | owner does not approve the privacy design, or confident-key precision < 0.99 | #401 stays parked |
| RQ-8 | a kind fails the agreement bars | the judge only ranks sheets for that kind; human review carries the check |
| RQ-9 | no arm beats union-find on chaining at matched recall | keep the current grouping; ship only N-1 (the job move) |
| RQ-10 | PE frame vectors lose by more than 0.02 | close #435 as "whole-image search stays in the core API" |
| Fine-grained | RQ-3 and RQ-4 both stop | make and model stays parked under #346; no adapter work |

### 7.2 Risks

| Risk | Mitigation |
|---|---|
| A held-out half is read too often and stops being held out | one read per method family; fresh set for acceptance (4.2) |
| The B5 crop path changes the vectors the research measured | E-3 and later run on the merged B5 stack only; a parity check of a 1,000-crop sample against the stored vectors first |
| Open Images train split lacks enough open-set crops | the fetcher counts before downloading images; the quota is registered; INCONCLUSIVE ends the line (7.1) |
| Host and GPU contention with peer sessions | windows announced and agreed; one heavy run at a time; pause at once when asked |
| leidenalg licence | OQ-4; scikit-learn fallback; the method is the same family |
| Judge bias or drift | agreement per kind, planted controls, blind human re-judge, prompt versioned |
| Biometric and plate leakage | never to a hosted model; local-only artefacts; refusal tests; purge design before any plate work |
| Result looks good on one stratum and hurts another | stratum bars in 4.6 |
| Scope creep into a vehicle-only product | every class word is project data; Open Images animals run as a non-vehicle set in E-1 and E-5 |
| Research data lost | backing JSON in `docs/benchmarks/` with checksums; never pruned |

### 7.3 Open questions for the owner

| ID | Question | Recommendation |
|---|---|---|
| OQ-1 | Faces have no public identity set with a clear licence. May aggregate face numbers from LFW, CALFW and CPLFW be published, or do published face claims stay limited to Open Images faces checked by a person? | publish aggregates from LFW-family sets with a licence caveat; per-image data never |
| OQ-2 | Approve three research GPU windows after the v0.7.0 tag (about 5 h, 13 h and 6 h) on GPU 2 or GPU 0? | yes, one at a time, announced to peers |
| OQ-3 | Haiku budget: at most about 300 sheets for E-10 and 50 per other experiment? | yes |
| OQ-4 | Is leidenalg (GPL-3.0) acceptable as a product dependency in the AGPL repo if Leiden wins? | decide only if E-5 picks Leiden; otherwise moot |
| OQ-5 | Should N-1 (person grouping as a Celery job) join the program now (a post-B2 batch) rather than wait for research? | yes; it is a correctness fix, not research |
| OQ-6 | Is #401 (plate identity) still wanted, given that public data cannot test plate linking and the privacy design comes first? | keep parked; run E-9 read quality only if the owner wants the feature this year |
| OQ-7 | Owner time: about 8 hours for human checks over the three windows. Acceptable, or name a delegate? | acceptable, spread over the windows |
| OQ-8 | Which winners should the owner repeat on the private photo set before adoption? | RQ-1, RQ-3 and RQ-4 winners |
| OQ-9 | Near-duplicate defaults: one global default per path, or a project setting? | a default per path; a setting only if E-2 shows the best value moves between sets |
