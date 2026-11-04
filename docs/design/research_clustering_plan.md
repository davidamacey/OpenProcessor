# Clustering for curation and the visual judge: research plan and rebuild seams

| Field | Value |
|---|---|
| Superseded parts | the landing lanes, waves and file ownership of this plan's section 8, its GPU lock and its GPU-hour estimates are superseded by the program (review RR-16, owner answers of 2026-11-04): lanes in [program/03b](program/03b_b2_w2_job_families.md) (W2.3a and W2.3b), [program/03d](program/03d_b2_s2_control_data.md), [program/03c](program/03c_b2_w3_beat_reaper_workers.md), [program/05](program/05_b4_stateless_api.md) and [program/03g](program/03g_w5e_plate_text.md); one lock `gpu<N>.lock` per card shared by every topic ([program/00b](program/00b_traceability.md) X-14); short tiered runs under a cap of about 10 GPU hours before the tag ([program/00e_research_track.md](program/00e_research_track.md) section 2.1). A fresh agent loads those files first |
| Status | plan, research. Nothing here is implemented or measured anew. Written 2026-11-04 against `origin/main` at `497315954` |
| Scope | grouping of curation items and region boxes (class level), faces (identity level) and whole images (scene level); detector-class roots as a partition source; graph, linkage, density and community methods; incremental against batch; versioned partitions with stable ids; GPU clustering; the visual judge, its calibration and the blind human re-judge |
| Companion file | [research_clustering_plan_b_protocol.md](research_clustering_plan_b_protocol.md): the prototype protocol (section 6 of this plan), the experiments, the judge procedure in full, GPU etiquette and stop rules. This file holds what is decided or to be built; file b holds how it is measured |
| Absorbs | the umbrella plan of #346 and the judge plan of #406 (both former `docs/design/issue_plans/` files, deleted in this commit; git history keeps them). Their corrections C1-C15, failure register and judge design are folded into sections 2, 4, 9 and file b |
| Depends on | the PostgreSQL + Celery + Valkey foundation ([02](program/02_control_plane_database.md), [02a](program/02a_postgres_schema.md), [02b](program/02b_jobs_locking_events.md)): long jobs are Celery tasks with `jobs` rows; OpenSearch keeps search and vector data. Research experiments run after the v0.7.0 tag (MQ-9); the seams in section 5 are proposed for the rebuild itself |
| Issues | tracker #346; #406 (judge); #398 (detector-class roots, design owned by the auto-label research plan); interfaces to #416 (near-duplicates), #347 and #365 (encoders), #400 (attribute signals), #401 (plate text), #435 and #436 (frame vectors and group-by-image) |
| Reader | a fresh agent with no memory of this session. Every proposed table, field, kind and route has a name, a type and an owning plan file and lane |

## 1. Plain summary

**What runs today.** Four separate grouping mechanisms share the word "cluster" and nothing else
([cluster_types.md](../research/cluster_types.md)):

- curation items: FAISS spherical k-means with K = min(512, n // 6) over PE-Core-L14-336 (1024-d),
  then an on-demand complete-linkage split of one cluster at cosine distance 0.25;
- curation region boxes: MiniBatchKMeans with k = max(8, round(n / 800)), the opposite extreme
  (about 6 crops per item cluster against about 800 boxes per box cluster);
- faces: connected components (single linkage, union-find) at cosine 0.7 over ArcFace, exact and
  O(n^2), inside a request (`POST /persons/cluster`), never scored;
- whole images: FAISS IVF cells over MobileCLIP, which are an index, not clusters.

**What the research found.** Most fragmentation is configuration, not method: IVF at n // 6 forces
pair recall near zero and depends on row order (ARI 0.306 under a permutation). Above pair recall of
about 0.3 the embedding neighbourhoods are the limit (P@10 = 0.82). Every challenger method failed a
pre-registered bar on real data or could not be read. The one lead is in the gate, not the method:
one root per detector class needs about a third of IVF's labelling decisions at the same precision
(Z4), but its open-set read is pending (#398). Graph refinement (Leiden CPM) works on synthetic
vectors and is unproven on real crops. The visual judge (Haiku on contact sheets) passes its bars on
COCO crops only; the local 4B VLM is not a judge; the layout calibration was vacuous; the blind human
re-judge has never been done.

**What this plan recommends.**

1. **Separate three concerns the code mixes today.** (a) The *index and assigner* (IVF centroids,
   OpenSearch kNN) that places a new item quickly. (b) The *partition*: an immutable, versioned
   result of one clustering run, with stable cluster identities carried across runs. (c) The
   *review state*: human names, anchors, merges, splits and constraints that every later run must
   keep. Today all three are the `cluster_id` integer on items, overloaded with the class id
   (`cluster_id == class_id` below 10000, `RESIDUAL_CLUSTER_ID_OFFSET`).
2. **Build the partition foundation in the rebuild, with the methods that ship today.** PostgreSQL
   records every clustering run as a partition (method, space, model, parameters, calibration,
   metrics, lineage) with a membership snapshot; an OCC-guarded activation pointer decides which
   partition is served; a projection job writes the served assignment onto items in OpenSearch.
   IVF, region k-means and person grouping all write through it from day one, so the tables are
   used, not speculative, and a research winner later is a new method value in a revisioned
   `cluster_profile` config, not a schema change. Section 5 has the exact tables, kinds, fields and
   routes.
3. **Run the research as pre-registered, stop-ruled prototypes after the tag**, in tracked
   `scripts/research/clustering/` code with exact pins, on public labelled data (COCO, Open Images,
   CUB-200, SOP, LFW family) plus set P (owner-run, results private), with a calibrated visual judge
   and a blind human re-judge on every quality claim (file b).
4. **Faces are first-class.** The first face experiment compares union-find 0.7 with an Immich-style
   core-point rule, HDBSCAN on the identity space, graph HAC with the same-image cannot-link and
   Leiden CPM, plus calibrated incremental assignment with abstention.
5. **GPU clustering is agreement-gated.** The GPU builds the kNN graph (cuVS or FAISS GPU); the
   clustering objective stays the CPU method unless a GPU method agrees with it on real COCO data.
   A deployment records which backend version passed, and a run refuses an unvalidated GPU backend.

## 2. What is measured today

Every figure is quoted from the linked note, which holds the method and limits. Every run had a
loaded host (1-minute load 9 to 90), so only within-table timings compare. [R] = real data,
[S] = synthetic (speed and scale only, never quality).

### 2.1 Production partitions

| Kind | What ships | Measured | Source |
|---|---|---|---|
| Items, 3,000-item public set (944 images, 3,961 crops, 2,950 labelled, 86 classes) | IVF K = min(512, n // 6) | held-out pair precision 0.716, pair recall 0.027, BCubed P 0.784, R 0.186 [R]; on the full set pair P 0.696, R 0.022, false merge 0.304 (0.400 low-res), false split 0.978; strata P: low-res 0.601, mixed scene 0.723, near-dup 0.896, rare 0.944, similar classes 0.882; 20.8 % of items outside their cluster's majority class; ARI 0.306 under row permutation, 0.356 across a re-ingest | [cluster_eval_baseline.md](../research/cluster_eval_baseline.md), [clustering_research_consolidated.md](../research/clustering_research_consolidated.md) 3.1, `docs/benchmarks/cluster_eval/score.json` |
| Items, 200-image COCO project (1,364 crops, 539 labelled, 11 classes) | same | pair P 0.905, R 0.088, 128 clusters; stability ARI against a second project 0.369 [R] | [leiden_refinement_prototype.md](../research/leiden_refinement_prototype.md) 3b |
| Items, earlier 350k-crop corpus | IVF (UMAP + AHC retired) | UMAP to 50-d + AHC put 97.7 % in one cluster; HDBSCAN gave one cluster (95-100 %) or 100 % noise over `min_cluster_size` 5-200; sklearn complete-linkage AHC at 347k ran about 1 h and peaked near 124 GB; IVF largest bucket under 1 %, median about 96 items at 128k; GPU k-means about 12 s at 50k, 30 s at 128k [R] | [clustering_methods.md](../research/clustering_methods.md) 2 |
| Items, 2,469-item ablation | HDBSCAN, AHC, OPTICS after UMAP | ward AHC auto-n 15 clusters, coherence 0.35, OTP 57.15; best HDBSCAN (H5) OTP 6.15, 5 % noise; OPTICS 23 % noise; fixed-threshold cosine AHC collapsed to one cluster [R, small] | [clustering_methods.md](../research/clustering_methods.md) 8 |
| Region boxes | MiniBatchKMeans k = max(8, round(n / 800)) | never scored | [cluster_types.md](../research/cluster_types.md) |
| Faces (core) | union-find at cosine 0.7, exact FAISS range search | never scored; about 4 min at 200k faces, about 1.5 h at 1M, inside the request | `src/services/face_identity/person_grouping.py`; #346 comment of 2026-11-01 |
| Whole images (core) | FAISS IVF, 1,024 cells (`global`), 2,048 for faces | never scored as clusters | [cluster_types.md](../research/cluster_types.md) |

### 2.2 Cost of the IVF pass (set P and synthetic, CPU FAISS 1.15.1)

| Crops | Wall s | Fetch s | Fit s | Write-back s | Peak RSS MiB |
|---|---|---|---|---|---|
| set P 10,000 / 25,000 / 50,000 / 91,184 | 22.0 / 80.6 / 203.2 / 231.2 | 2.5 / 6.3 / 10.2 / 17.4 | 15.5 / 67.0 / 177.5 / 181.6 | 4.2 / 8.5 / 15.4 / 30.4 | 1,043 / 1,554 / 2,057 / 3,002 |
| synthetic 250k / 500k / 1M [S] | 203.5 / 322.6 / 596.0 | 40.1 / 78.8 / 167.9 | 84.6 / 88.7 / 99.9 | 79.1 / 155.7 / 336.2 | 6,382 / 11,278 / 21,120 |

Fetch is 0.16-0.25 ms per crop and write-back 0.31-0.42 ms per crop, flat from 10k to 1M; the fit is
super-linear below the 50,000-vector training cap and flat above it; pass overhead about 8 KB per
crop. Source: [v070_measurement_log.md](../research/v070_measurement_log.md).

### 2.3 Geometry of the PE crop space (real COCO crops, set P crops for fitting)

| | COCO (1,364) | set P (20,000 sample) |
|---|---|---|
| mean random-pair cosine (5th-95th) | 0.50 (0.30-0.72) | 0.63 (0.45-0.80) |
| effective rank of 1,024 | 122 | 104 |
| median 1-NN / 10-NN cosine | 0.87 / 0.77 | 0.92 / 0.88 |
| hubness skew of N10, largest N10 | 2.57, 89 | 2.42, 119 |

Neighbour purity (COCO, labelled neighbours only): P@1 0.877, P@10 0.821; P@10 0.781 for crops under
32 px, 0.741 for crops longer than 2:1, 0.871 at side 96 px and above. 21.1 % of crops are longer
than 2:1; the median centre-crop keeps 0.69 of the long side; the detector class agrees with the
COCO label on 92.1 % of crops [R]. Oracle-threshold curves (best precision at recall at least
0.2 / 0.3 / 0.5): average-linkage AHC 0.920 / 0.856 / 0.404; complete 0.878 / 0.730 / 0.301; Leiden
CPM 0.925 / 0.889 / 0.776. Held-out average linkage (two folds): 0.796-0.958 precision at recall
0.11-0.37. Source: [clustering_research_consolidated.md](../research/clustering_research_consolidated.md) 4.1.

### 2.4 Decisions per arm

| Arm | Decision | Key number | Source |
|---|---|---|---|
| IVF as the cluster definition | stays (no challenger passed) | best on `rare`; see 2.1 | [cluster_program_outcome.md](../research/cluster_program_outcome.md) 3 |
| Average linkage inside the detector class | parked | fresh set Y1 (3,355 crops, 2,334 labelled): A1 0.927, A2 0.909, A3 0.951, A5 0.128, A7 1.000 pass; A4 0.288 against 0.329 and A6 -0.054 fail; cut 0.346 / 0.345 / 0.515 across sets | [cluster_config_fresh_set.md](../research/cluster_config_fresh_set.md), [cluster_config_decision.md](../research/cluster_config_decision.md) |
| Label-free per-key cut (alpha 1.175) | killed | held-out BCubed R 0.300 against 0.374, pair R 0.063 against 0.087 | [cluster_config_fresh_set.md](../research/cluster_config_fresh_set.md) Y2 |
| Whitening, k-reciprocal, DBA | killed | precision at matched recall -0.033 / -0.065, -0.014, -0.011 | [cluster_config_experiments.md](../research/cluster_config_experiments.md) X3 |
| Novelty flag by distance | killed | AUROC 0.838 and 0.851 against 0.90; best flag precision 0.733 / 0.761 | same, X4 |
| Judge-gated merge (Z3) | killed | +2.2 % effort against a 10 % saving bar | [cluster_config_loop_cost.md](../research/cluster_config_loop_cost.md) |
| Split by descent | killed | 10 of 14 (71 %) and 1 of 11 (9 %) impure clusters fixed; bar 80 % | [cluster_config_fresh_set.md](../research/cluster_config_fresh_set.md) Y4 |
| Loop roots, W0e (33,580 crops, 7,476 labelled held out) | direction | U1 / U2: IVF 0.776 / 0.988; one cluster per key 0.374 / 0.989 (15.8 open-set items per run, all wrong); within key 0.326 / 0.996 | [cluster_config_decision.md](../research/cluster_config_decision.md) 0A.1 |
| Loop roots, Z4 (7,209 labelled held out, 395 open-set) | INCONCLUSIVE | `key` U1 0.293, U2 0.982, open-set 0.733 on 19.9 per run (9.05 wrong); `key_guard` 0.377, 0.988, 0.976 on 9.2 (0.25 wrong); `within_key` 0.526 FAIL (813 roots); `ivf_loop` 0.834 / 0.979; `ivf_gate` 0.840 / 0.981; a 10 % noisy labeller drives U1 to 0.94-0.97 | [cluster_config_z4.md](../research/cluster_config_z4.md) |
| Leiden CPM, synthetic 20k / 100k / 500k (128-d) [S] | not evidence | pair recall 0.08 to 0.79-0.86; + CHAMELEON precision 0.953-0.954; modularity false merge 0.236 / 0.519 / 0.609; consensus 5x time for equal quality; HDBSCAN leaf 80 % noise; FINCH level 0 / 1 recall 0.40-0.49 / 0.95-0.96 with false merge 0.17-0.24 | [leiden_refinement_prototype.md](../research/leiden_refinement_prototype.md) 3 |
| Leiden CPM, 539 real COCO crops | parked | init g 0.1: 0.905 / 0.234; + CHAMELEON (RI 0.5, RC 0.95, tuned in-sample) g 0.03: 0.944 / 0.195, 9.8 % unassigned, stability 0.974; plain g 0.01: 0.745 / 0.524; modularity 0.709 / 0.579; HDBSCAN 76.6 % noise | same, 3b |
| Leiden at 1024-d [S] | calibration problem | best gamma moves from about 0.003 (128-d) to 0.001 or less; 500k: 380-450 s, 5.1 GB; kNN dominates (219-282 s IVF kNN); CHAMELEON 76.7 s on 20k single-item atoms | same, 4 |
| GPU arms (A6000, 20k, 128-d) [S] | parked | cuGraph Leiden (modularity) 0.776 / 0.910; cuGraph ECG 0.953 / 0.899, 7.2 s, 1.8 GB; cuML HDBSCAN 500k in 20.1 s, about 70 % noise; nothing above 20k for Leiden, ECG or kNN | same, 3c |
| Encoder swap | not adopted | WB2 R@1 (CARS196 / CUB / SOP): PE-Core-L 0.961 / 0.867 / 0.796; PE-Core-G 0.969 / 0.907 / 0.820 (about 8x encode, 5x memory); MobileCLIP2-S2 0.964 / 0.752 / 0.674; SigLIP 2 0.975 / 0.744 / 0.790; DINOv2-L 0.825 / 0.892 / 0.576; sanity check not met, exploratory | [wb2_standard_benchmarks.md](../research/wb2_standard_benchmarks.md) 9 |
| Near-duplicates | open (#416, owned elsewhere) | core MobileCLIP 0.99 catches 34 of 100 JPEG q60 copies and 13 of 100 half-size copies; 0.95 catches nearly all but mixes distinct images | [retrieval_baseline.md](../research/retrieval_baseline.md) |

### 2.5 The visual judge (E0)

| | Haiku COCO | Haiku OI vehicles | Local 4B COCO | Local 4B vehicles | Local 4B faces |
|---|---|---|---|---|---|
| Cluster sheets | 55 | 55 | 210 | 217 | 220 |
| Accuracy (Wilson 95 %) | 0.855 (0.738-0.924) | 0.764 (0.637-0.856) | 0.700 (0.635-0.758) | 0.645 (0.579-0.706) | 0.755 (0.694-0.807) |
| Kappa with truth | 0.697 | 0.298 | 0.400 | -0.009 | 0.519 |
| False "same" / false "different" | 0.094 / 0.217 | 0.000 / 0.765 | 0.283 / 0.317 | 0.007 / 1.000 | 0.470 / 0.000 |
| Flag precision / recall | 0.827 / 0.838 | 0.454 / 0.979 | 0.167 / 0.735 | 0.110 / 0.963 | 0.298 / 0.578 |
| Purity truth vs judge | 0.888 vs 0.886 | 0.890 vs 0.763 | 0.914 vs 0.620 | 0.902 vs 0.143 | 0.915 vs 0.835 |
| Pair kappa (decided rows) | 0.821 | 0.611 | 0.031 | 0.000 | 0.051 |
| Bars (pair kappa 0.7, flag precision 0.8) | pass | fail | fail | fail | fail |

Layout calibration: Haiku said "different" to all 50 vehicle sheets, so the rule picked 3 x 3
vacuously; the local 4B on COCO (30 clusters x 5 layouts) gave kappa 0.52 (3 x 3), 0.76 (4 x 4),
0.67 (6 x 4 at 128), 0.76 (6 x 4 at 160), 0.59 (6 x 4 at 192). Judge-vs-judge kappa on shared sheets
0.312 (COCO) and 0.000 (vehicles). The second local catalog entry was truncated at 600 output tokens
(255 of 259 vehicle replies invalid); three reruns at 4,096 tokens used 762-1,062 tokens and were
valid. The six Haiku subagents used about 580k tokens for 184 sheets. FG-0 (make and model labels,
machine review only): decided-row model accuracy 0.971, 29.6 % unsure, a quarter of same-make plants
accepted: INCONCLUSIVE. Sources: [visual_judge_e0.md](../research/visual_judge_e0.md),
[fine_grained_benchmark_results.md](../research/fine_grained_benchmark_results.md).

Code causes of the vacuous calibration (verified on `int/b1-correctness` at `81c4e4ef` by the former
#406 plan): `make-calibration` draws clean clusters with probability 1/5 (`n_bad` from
`(0, 2, 3, 5, 8)` in `scripts/research/visual_judge.py`), and `recommend_layout` in
`scripts/research/judge_agreement.py` has no guard for a judge that answers one way, so kappa 0 at
every layout picks the smallest. The judge exchange directory names sheets by id (`rc-ctl*` controls,
`*-r` twins, `cal*-<layout>`), which unblinds any reader.

### 2.6 What is not established

Open-set behaviour of class roots (Z5); any held-out real-data win for a graph method; anything above
20k on the GPU or at 1M with quality; any face grouping score; any region-box score; any whole-image
grouping score; viewpoint behaviour; the judge's agreement with a person; timings on a quiet host.

## 3. Literature and industry review

Tags: [L] read at the URL this session or in the cited internal note; [R] measured here.

### 3.1 Methods

| Family | Source | What it gives | Fit here |
|---|---|---|---|
| Leiden community detection, CPM objective | Traag, Waltman, van Eck 2019, [Sci. Rep.](https://www.nature.com/articles/s41598-019-41695-z); [leidenalg reference](https://leidenalg.readthedocs.io/en/stable/reference.html) (read 2026-11-04) | well-connected communities, a resolution knob, `CPMVertexPartition(initial_membership=..., weights=..., resolution_parameter=...)`, `Optimiser.optimise_partition(..., is_membership_fixed=[...])` pins nodes, CPM "well-defined for both positive and negative edge weights" (cannot-link as negative edges), `resolution_profile` for a gamma sweep | the scalable class-level arm; CPU only |
| cuGraph Leiden | [cuGraph docs](https://docs.nvidia.com/cugraph/latest/api_docs/api/cugraph/cugraph.leiden/) (read 2026-11-04) | `leiden(G, max_iter=100, resolution=1.0, random_state=None, theta=1.0)`, "the modularity optimizing partition", no initial partition | not CPM; contrast arm only |
| ECG | Poulin and Theberge 2018, [arXiv 1809.05578](https://arxiv.org/abs/1809.05578); in cuGraph | ensemble of level-1 Louvain runs | GPU contrast arm |
| CHAMELEON | Karypis, Han, Kumar 1999, [IEEE Computer](https://doi.org/10.1109/2.781637) | merge test by relative interconnectivity and closeness | post-filter, only if it adds held-out precision |
| Graph HAC at scale (ParHAC, TeraHAC) | [TeraHAC arXiv 2308.03578](https://arxiv.org/abs/2308.03578); code in [google/graph-mining](https://github.com/google/graph-mining) and [ParAlg/ParHAC](https://github.com/ParAlg/ParHAC) | (1+epsilon)-approximate average-linkage HAC on sparse graphs; trillion edges | the scalable form of average linkage, but C++ and Bazel only: no maintained Python package was found (searched 2026-11-04) |
| HDBSCAN | Campello, Moulavi, Sander 2013; [scikit-learn HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html); [cuML](https://docs.rapids.ai/api/cuml/stable/) | density clusters, noise label -1 | fails on PE crop space (no density gaps) [R]; plausible on an identity-trained face space |
| FINCH | Sarfraz, Sharma, Stiefelhagen, CVPR 2019 | parameter-free first-neighbour hierarchy | no granularity control [R] |
| Incremental Leiden (HIT-Leiden) | [arXiv 2601.08554](https://arxiv.org/abs/2601.08554), revised 2026-06 | incremental maintenance of Leiden communities | research code only; an outside review could not verify the code; not before a Leiden adoption |
| Rank-order clustering | Otto, Wang, Jain, [arXiv 1604.00989](https://arxiv.org/abs/1604.00989) | face clustering of 123M images; a per-cluster quality score for review | no maintained library |
| Learned face linkage (GCN-V/E, STAR-FC, Ada-NETS, Hi-LANDER) | [arXiv 2103.13225](https://arxiv.org/abs/2103.13225), [arXiv 2202.03800](https://arxiv.org/abs/2202.03800), [arXiv 2107.01319](https://arxiv.org/abs/2107.01319) | pairwise F about 0.92 on MS1M 584K splits (authors' figures) | trained on MS-Celeb-1M, which was withdrawn; a licence-clean training set is an owner decision |
| Hierarchical k-means for balanced curation | Vo et al. 2024, [arXiv 2405.15613](https://arxiv.org/abs/2405.15613), [code](https://github.com/facebookresearch/ssl-data-curation) | balanced topic tree over millions of images | whole-image browsing (K1) |
| Cluster transitions over time (MONIC) | Spiliopoulou et al., KDD 2006, [doi 10.1145/1150402.1150491](https://dl.acm.org/doi/10.1145/1150402.1150491) | survive, split, absorb, emerge, disappear, matched by member overlap | the id carry-over rule of a versioned partition |
| Optimal one-to-one matching | [scipy.optimize.linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) | Hungarian matching of old to new clusters on a Jaccard matrix | id carry-over, deterministic |
| Online assignment with abstention, microclusters | DenStream (Cao et al., SDM 2006); Immich deferred faces (below) | assign to a nearby cluster or hold as a microcluster until promoted | incremental assignment bands |
| Hubness and its reduction | Radovanovic et al. JMLR 2010; Feldbauer and Flexer 2019 ([KAIS](https://link.springer.com/article/10.1007/s10115-018-1205-y)) | scaling methods help kNN; centring only sometimes | measured: no clustering gain on crops [R] |

### 3.2 Evaluation

| Topic | Source | Rule taken |
|---|---|---|
| BCubed (headline partition metric) | Amigo et al. 2009, [doi 10.1007/s10791-008-9066-8](https://doi.org/10.1007/s10791-008-9066-8) | headline P, R, F; an abstained item counts as a singleton (harness rule) |
| V-measure | Rosenberg and Hirschberg 2007, [ACL D07-1043](https://aclanthology.org/D07-1043/); scikit-learn `homogeneity_completeness_v_measure` | reported beside BCubed |
| AMI | Vinh, Epps, Bailey 2010; scikit-learn `adjusted_mutual_info_score` | reported; chance-corrected |
| Cohen's kappa and its prevalence paradox | Feinstein and Cicchetti 1990; Byrt, Bishop, Carlin 1993 (PABAK) | a judge with one answer gives kappa near 0 against a skewed truth even at high accuracy; report raw agreement, kappa and PABAK (2 x agreement - 1) together, and balance calibration sets |
| Prediction-powered inference | Angelopoulos et al., Science 2023, [arXiv 2301.09633](https://arxiv.org/abs/2301.09633); [ppi_py](https://github.com/aangelopoulos/ppi_py) (MIT) | combine many judge verdicts with a small human-labelled subset to get a bias-corrected purity estimate with a valid interval; the human re-judge becomes the rectifier, not only a kappa check |
| LLM-as-judge biases | Zheng et al. 2023, [arXiv 2306.05685](https://arxiv.org/abs/2306.05685); [MLLM-as-a-Judge](https://arxiv.org/abs/2402.04788) | position bias (re-render in a second order), planted controls, prompt versioned |
| Intervals | [statsmodels proportion_confint](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html) (Wilson), [scipy.stats.bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) (paired) | Wilson for rates, bootstrap over images or sheets for deltas |

### 3.3 Libraries (versions read on PyPI 2026-11-04)

| Library | Version | Licence | Use |
|---|---|---|---|
| scikit-learn | 1.9.1 (Python >= 3.11) | BSD-3 | `AgglomerativeClustering(connectivity=..., distance_threshold=...)`, `HDBSCAN`, metrics, `IsotonicRegression` |
| faiss-cpu | 1.15.1 | MIT | exact and IVF kNN, CPU range search |
| FAISS GPU | `faiss-gpu-cu12` (no cu13 wheel on PyPI) | MIT | GPU flat and IVF kNN; `k` and `nprobe` capped at 2048; no GPU range search ([wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU), [issue 565](https://github.com/facebookresearch/faiss/issues/565)) |
| cuVS (`cuvs-cu13`) | 26.10.0 | Apache-2.0 | `cuvs.neighbors.all_neighbors.build(dataset, k, params)` with `brute_force`, `ivf_pq` or `nn_descent` and batching by clusters for graphs larger than one GPU ([docs](https://docs.nvidia.com/cuvs/api-reference/python-api-neighbors-all-neighbors)) |
| cuML (`cuml-cu13`), cuGraph (`cugraph-cu13`) | 26.10.0 (released 2026-10-08) | Apache-2.0 | HDBSCAN, ECG, connected components; `AgglomerativeClustering` is single linkage only, no threshold ([gpu_clustering.md](../research/gpu_clustering.md)) |
| igraph, leidenalg | 1.0.0, 0.12.0 | GPL-2.0-or-later, GPL-3.0-or-later | Leiden CPM |
| ppi-python | 0.2.3 | MIT | prediction-powered intervals |
| statsmodels, scipy | pinned at R0 | BSD-3 | Wilson, bootstrap, Hungarian |

Licence note: igraph and leidenalg are GPL; combining GPLv3 code with AGPLv3 is allowed by GPLv3
section 13, but a product dependency needs the owner's yes and an `ATTRIBUTION.md` row (J5). Until
then they are research dependencies only.

### 3.4 Industry and open-source practice

| Product | What it does | Reported quality | Lesson |
|---|---|---|---|
| Immich ([facial recognition docs](https://docs.immich.app/features/facial-recognition), read 2026-11-04; [config docs](https://docs.immich.app/install/config-file/)) | same `buffalo_l` models as ours; per face, search within a maximum recognition distance (0.3-0.7 advised; default 0.5 cosine distance per the config fixtures), at most a few neighbours; join the nearest face's person, else create a person if the face is a core point (minimum recognised faces, default 3, "including the face being considered"), else defer to the end of the queue and then treat as an outlier; a nightly job retries unassigned faces; a full re-run "does behave like DBSCAN" | none published; its tracker reports look-alike merges, chained "worm" clusters across age, no split tool, and manual merges lost on a forced re-run ([cluster_search_makemodel_plan.md](cluster_search_makemodel_plan.md) 3.1) | core points, deferral and incremental assignment are the baseline; we must add stable ids, anchors, split and suggestions |
| PhotoPrism ([face docs](https://docs.photoprism.app/developer-guide/vision/face-recognition/)) | DBSCAN over face embeddings, core 4-5, epsilon "calibrated per face model" | none | thresholds do not transfer between models |
| FiftyOne Brain ([docs](https://docs.voxel51.com/brain/index.html), read 2026-11-04) | uniqueness, similarity index, near and exact duplicates, representativeness, leaky splits, mistakenness, hardness, UMAP/t-SNE/PCA visualization; no clustering method in the Brain | none | the curation tools that matter are similarity, duplicates, leakage and representativeness, all kNN based; clustering is a navigation aid, and visualization never decides membership (agrees with [clustering_methods.md](../research/clustering_methods.md) 7) |
| cleanlab Datalab ([issue types](https://docs.cleanlab.ai/stable/cleanlab/datalab/guide/issue_type_description.html)) | near-duplicate = nearest-neighbour distance below a fraction of the dataset's median NN distance; outlier score from kNN distance | none | thresholds relative to the space's own NN distribution, the same lesson as 2.3 (anisotropy) |
| CVAT, Label Studio | no clustering; Label Studio can collect human judgements (image + choices) | - | a candidate tool for the human re-judge (J12), not for clustering |
| Apple Photos ([article](https://machinelearning.apple.com/research/recognizing-people-photos)) | progressive gallery of known people; observations assigned or left unassigned | partly read | "leave unassigned" is a first-class state |

**Common pattern** in production photo apps: a core-point rule, incremental nearest-person
assignment with a distance cap, an outlier state, thresholds tied to one model, and no published
accuracy. These are density methods that work on identity embeddings (which have gaps) and fail on
general semantic embeddings (which do not). Nobody publishes a stable-id or correction-preserving
design; that is ours to get right.

## 4. Candidate methods

✅ adopt (design or measured), ❌ reject (evidence or constraint), 🟡 research arm (needs a read
on fresh real data). "Owner" names the plan that runs the experiment.

### 4.1 Class level: curation items and region boxes

| Method | Verdict | Reason |
|---|---|---|
| IVF k-means as the cluster definition (K = n // 6) | ❌ as clusters | configuration artefact: K fixed by pool size, pair recall near 0, ARI 0.306 under permutation; stays only until a winner passes |
| IVF centroids as the coarse index and ingest-time assigner | ✅ | O(1) assignment, linear pass (2.2); K chosen for the index, not for clusters |
| Order independence (sort input by id, seed from a stable key) | ✅ | required for any method; #372; permutation ARI must be 1.000 |
| Global average-linkage HAC on a mutual 30-NN graph, dev-calibrated cut | 🟡 lead at <= 150k | best real-data curve at recall 0.2 (0.920) and equal to tuned Leiden with nothing unassigned; scale path open: sklearn needs bounded blocks (124 GB at 347k complete linkage), ParHAC/TeraHAC have no Python package (J6) |
| Leiden CPM on the mutual 30-NN graph (leidenalg), gamma log-grid calibrated on dev | 🟡 lead at scale | keeps precision at high recall (0.776 at 0.5); 500k 1024-d in 380-450 s; pins anchored nodes (`is_membership_fixed`); cannot-link as negative edges; GPL (J5) |
| + CHAMELEON-style filter | 🟡 | only if it adds held-out precision; needs a faster implementation for many atoms |
| Detector-class roots, with and without the hold-back guard (#398) | 🟡 lead for the loop | Z4 effort 0.35x IVF; open-set unread. Owned by the auto-label research plan; this plan supplies the partition seam (`method = detector_class`) and the harness |
| Detector class as a soft prior (cannot-link between confident different classes) | 🟡 | free constraint, 92.1 % agreement on all crops; must be measured on the unconfident residual subset (consolidated E5, never run) |
| Within-class average linkage | ❌ parked | A4 and A6 fail on a fresh set; the cut moves between sets |
| Modularity Leiden (CPU or cuGraph) | ❌ | over-merges, worse with n (false merge 0.236 to 0.609) |
| Consensus Leiden | ❌ | 5x time, equal quality |
| HDBSCAN (any library), FINCH, OPTICS, UMAP before clustering | ❌ | 77-100 % noise or no granularity control on real crops; UMAP collapses geometry |
| Whitening, k-reciprocal, DBA, CSLS | ❌ | X3 failed its bar |
| Novelty-by-distance flag, split by descent, judge-gated merge | ❌ | bars failed (2.4) |
| Learned clustering heads (SCAN, TEMI, TURTLE) | ❌ | assume known K and balanced classes |
| HIT-Leiden incremental maintenance | ❌ for now | unverified research code; revisit only if Leiden is adopted and a full re-run misses its time budget |
| Multi-prototype clusters (up to 8 medoids) and nearest-prototype assignment | 🟡 | the domain-agnostic answer to viewpoint modes; untested on real data (CL-7) |
| Region boxes: the same arms per box space | 🟡 | today's k = n / 800 was never scored; needs a box-level labelled set (CL-3) |

### 4.2 Identity level: faces

| Method | Verdict | Reason |
|---|---|---|
| Union-find at cosine 0.7 | ❌ once an arm beats it on chaining | single linkage chains; O(n^2); never scored; the speech project's same mistake (cross-domain note 1) |
| Immich-style core-point incremental (distance 0.5, core 3, detection score 0.7) | 🟡 first baseline | same model weights; production-tested shape |
| HDBSCAN on ArcFace vectors (scikit-learn; cuML on GPU) | 🟡 | identity spaces have density gaps (unlike PE crops); cuML ran 500k in 20 s |
| Average or complete graph HAC on mutual kNN, same-image cannot-link | 🟡 | refuses bridges; complete linkage may suit identity (cross-domain note 1) |
| Leiden CPM with negative same-image edges | 🟡 | signed CPM is supported |
| Rank-order clustering | ❌ unless every library arm fails | no maintained library |
| Learned GCN linkage | ❌ | training data withdrawn; licence |
| Approximate kNN (FAISS IVF or HNSW, cuVS) instead of exact range search | 🟡 | the scale fix; measure recall against exact at 200k and 1M real faces |
| Incremental assignment to the nearest of up to 8 prototypes, accept / review / unassigned bands by isotonic calibration, margin rule | 🟡 bands, ✅ design | precision >= 0.99 at accept is the bar |

### 4.3 Scene level: whole images

| Method | Verdict | Reason |
|---|---|---|
| IVF cells as albums | ❌ as clusters | cells are an index |
| Hierarchical k-means tree (Vo 2024) for browsing | 🟡 low priority | balanced navigation; scored by label-set Jaccard purity (CL-8) |

### 4.4 GPU

| Method | Verdict | Reason |
|---|---|---|
| GPU kNN graph build (cuVS `all_neighbors`, FAISS GPU flat or IVF) feeding the CPU method | 🟡 lead | kNN dominates cost at 1024-d (219-282 s of 380-450 s at 500k); output equals CPU's when graph recall >= 0.99, so the agreement gate is cheap to pass |
| cuGraph ECG | 🟡 contrast | one 20k synthetic row |
| cuML HDBSCAN for faces | 🟡 | see 4.2 |
| cuGraph Leiden as CPM | ❌ | modularity only, no initial partition |
| cuML `AgglomerativeClustering` | ❌ | single linkage only, no threshold |
| GPU range search | ❌ | not in FAISS GPU; use kNN plus threshold |

### 4.5 Judge

| Method | Verdict | Reason |
|---|---|---|
| Haiku on COCO crop cluster sheets | ✅ authority for that kind, subject to the human re-judge | pair kappa 0.821, flag precision 0.827 |
| Haiku on vehicle-type sheets | ❌ until the human pass decides | labels may be the problem (false "different" 0.765) |
| Local 4B VLM | ❌ authority, ranking only | pair accuracy 0.51 |
| Second local catalog entry at 2,048 output tokens | 🟡 | never scored (truncation) |
| Kappa alone as the agreement statistic | ❌ | prevalence paradox; report raw agreement, kappa and PABAK |
| Prediction-powered purity estimate (judge + human subset) | ✅ | valid interval with a small human sample (3.2) |
| Faces, plates or identity text to a hosted judge | ❌ | privacy; refused by declared kind today (owner decision OI-2: no new check beyond that) |
| Set P sheets to a hosted judge | ❌ by default | only with the owner's yes per named sample (J8) |

## 5. Seams the v0.7.0 rebuild must include

Principle: one owner per value ([02](program/02_control_plane_database.md) section 2). PostgreSQL
owns the partition record, cluster identities, membership snapshots, human review state,
calibrations and validations. OpenSearch owns the **served** assignment on items and boxes as search
data (02 section 3.4), written by one projection path, exactly like the class-name projection
(D-47). Blobs (contact sheets, kNN graphs kept for audit) go behind the storage seam.

Two tiers, so nothing is dead schema:

- **S-core (v0.7.0 rebuild):** used from day one by the methods that ship (IVF, region k-means,
  person grouping). Replaces `last_clustering.json`, the region-cluster markers, the
  `cluster_id`/`class_id` namespace overload and the IVF retrain hidden inside `autolabel`.
- **S-later:** lands with the feature that first writes it (after a research PASS or with the
  Cropwright review screens). The DDL is given now so the later lane does not redesign it.

### 5.1 PostgreSQL tables (conventions of [02a](program/02a_postgres_schema.md) section 1: `uuidv7()` keys, `timestamptz`, `text` + `CHECK` with one Python source, RLS on `project_id`, `*_by` actor columns)

```sql
-- S-core. One row per clustering run's output. Immutable after status 'ready' except status.
CREATE TABLE cluster_partitions (
  id               uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id       uuid NULL REFERENCES projects(id) ON DELETE CASCADE,  -- NULL: deployment (core faces, core images)
  target           text NOT NULL CHECK (target IN ('item','region_box','face','image')),
  level            text NOT NULL CHECK (level IN ('class','identity','scene')),
  embedding_field  text NOT NULL,          -- 'pe_embedding', 'region_box_embeddings', 'face_embedding', 'global_embedding'
  embedding_model  text NOT NULL,          -- model name and engine version, e.g. 'pe-core-l14-336@<engine sha256[:12]>'
  preprocess_id    text NOT NULL,          -- crop preprocessing variant (owned by the embeddings plan)
  method           text NOT NULL CHECK (method IN ('ivf','minibatch_kmeans','union_find','detector_class',
                     'edit','graph_hac','leiden_cpm','core_point','hdbscan')),  -- research values added on PASS
  method_version   text NOT NULL,
  profile_config_id uuid NULL,             -- cluster_profile document and revision that drove the run
  profile_revision integer NULL,
  params           jsonb NOT NULL,         -- validated by the method's pydantic params model
  calibration_id   uuid NULL REFERENCES cluster_calibrations(id) ON DELETE RESTRICT,
  pool_filter      jsonb NOT NULL,         -- the residual gate as data
  input_digest     text NOT NULL,          -- sha256 of the sorted (item_key, vector version) list
  seed             bigint NOT NULL,
  backend          text NOT NULL CHECK (backend IN ('cpu','gpu')),
  backend_detail   jsonb NOT NULL,         -- library versions, GPU name; joins cluster_backend_validations
  parent_partition_id uuid NULL REFERENCES cluster_partitions(id) ON DELETE SET NULL,  -- the served one at start
  job_id           uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  status           text NOT NULL CHECK (status IN ('building','ready','failed','discarded')),
  n_items          integer NULL, n_clusters integer NULL, n_review integer NULL, n_unassigned integer NULL,
  metrics          jsonb NOT NULL DEFAULT '{}',   -- label-free: size histogram, dispersion, stability vs parent
  diff             jsonb NULL,                    -- MONIC transition counts against the parent, moved items, anchored moves
  stage_durations  jsonb NOT NULL DEFAULT '{}',
  peak_rss_mb      integer NULL, peak_vram_mb integer NULL,
  created_at       timestamptz NOT NULL DEFAULT now(),
  finished_at      timestamptz NULL,
  created_by       text NOT NULL,
  UNIQUE (id, project_id)                         -- target of the membership composite FK
);
CREATE INDEX ix_cluster_partitions_recent ON cluster_partitions (project_id, target, level, created_at DESC);

-- S-core. The served partition per (project, target, level, field). OCC on revision; history below.
CREATE TABLE cluster_activations (
  project_id       uuid NULL REFERENCES projects(id) ON DELETE CASCADE,
  target           text NOT NULL,
  level            text NOT NULL,
  embedding_field  text NOT NULL,
  serving_partition_id uuid NOT NULL REFERENCES cluster_partitions(id) ON DELETE RESTRICT,
  pending_partition_id uuid NULL REFERENCES cluster_partitions(id) ON DELETE RESTRICT,  -- being projected
  revision         bigint NOT NULL DEFAULT 1,
  updated_at       timestamptz NOT NULL DEFAULT now(),
  updated_by       text NOT NULL,
  UNIQUE NULLS NOT DISTINCT (project_id, target, level, embedding_field)
);
CREATE TABLE cluster_activation_events (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  project_id uuid NULL REFERENCES projects(id) ON DELETE CASCADE,
  target text NOT NULL, level text NOT NULL, embedding_field text NOT NULL,
  from_partition_id uuid NULL, to_partition_id uuid NOT NULL,
  reason text NOT NULL CHECK (reason IN ('auto','human','rollback','edit','upgrade')),
  at timestamptz NOT NULL DEFAULT now(), by text NOT NULL
);

-- S-core. Stable identity of a cluster across partitions (names, anchors, promotion live here once).
CREATE TABLE cluster_identities (
  cluster_uid      uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id       uuid NULL REFERENCES projects(id) ON DELETE CASCADE,
  target           text NOT NULL, level text NOT NULL,
  name             text NULL,
  name_source      text NULL CHECK (name_source IN ('human','import','vlm')),
  anchored_at      timestamptz NULL,               -- a person curated it: members pinned in later runs
  anchored_by      text NULL,
  promoted_class_id integer NULL,                  -- the project_classes id it was promoted to (by id inside the project)
  first_partition_id uuid NULL REFERENCES cluster_partitions(id) ON DELETE SET NULL,  -- history only
  last_partition_id  uuid NULL REFERENCES cluster_partitions(id) ON DELETE SET NULL,  -- the reaper drops an
                                                   -- identity no partition references unless it is anchored or named
  revision         bigint NOT NULL DEFAULT 1,
  CHECK ((anchored_at IS NULL) = (anchored_by IS NULL))
);

-- S-core. Per-partition view of a cluster (size, prototypes, lineage).
CREATE TABLE partition_clusters (
  partition_id     uuid NOT NULL REFERENCES cluster_partitions(id) ON DELETE CASCADE,
  cluster_uid      uuid NOT NULL REFERENCES cluster_identities(cluster_uid) ON DELETE CASCADE,
  project_id       uuid NULL,
  display_no       integer NOT NULL,               -- small number for the UI, per partition
  parent_uid       uuid NULL,                      -- hierarchy (a split's parent)
  transition       text NOT NULL CHECK (transition IN ('new','survived','split','merged','absorbed')),
  predecessor_uids uuid[] NOT NULL DEFAULT '{}',
  size             integer NOT NULL CHECK (size > 0),
  dispersion       real NULL,
  prototypes       jsonb NOT NULL DEFAULT '[]',    -- up to 8 {item_key, count, dispersion, mode_tag, mode_source}
  PRIMARY KEY (partition_id, cluster_uid),
  UNIQUE (partition_id, display_no)
);

-- S-core. Membership snapshot of one partition (bulk-written with COPY by the run).
CREATE TABLE cluster_memberships (
  partition_id     uuid NOT NULL,
  project_id       uuid NULL,
  item_key         text NOT NULL,                  -- crop_id; crop_id || ':' || box_id for boxes; face id
  cluster_uid      uuid NULL,                      -- NULL: unassigned
  state            text NOT NULL CHECK (state IN ('assigned','review','unassigned')),
  score            real NULL,                      -- calibrated similarity to the best prototype
  margin           real NULL,                      -- to the second-best cluster
  PRIMARY KEY (partition_id, item_key),
  FOREIGN KEY (partition_id, project_id) REFERENCES cluster_partitions(id, project_id) ON DELETE CASCADE,
  CHECK ((state = 'unassigned') = (cluster_uid IS NULL))
);
CREATE INDEX ix_cluster_memberships_cluster ON cluster_memberships (partition_id, cluster_uid);

-- S-core. Calibrated thresholds; valid only for their space, model and preprocessing. Immutable (trigger).
CREATE TABLE cluster_calibrations (
  id               uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id       uuid NULL REFERENCES projects(id) ON DELETE CASCADE,   -- NULL: deployment default
  target text NOT NULL, level text NOT NULL,
  embedding_field  text NOT NULL, embedding_model text NOT NULL, preprocess_id text NOT NULL,
  kind             text NOT NULL CHECK (kind IN ('cut','cpm_gamma','assign_bands','core_point')),
  fitted_on        text NOT NULL,      -- 'pin:<manifest_sha256>' or 'project:<id>@holdout:<freeze id>'
  body             jsonb NOT NULL,     -- thresholds, isotonic knots, bands; pydantic-validated
  dev_metrics      jsonb NOT NULL,     -- dev-split precision and recall at the chosen point
  created_at       timestamptz NOT NULL DEFAULT now(), created_by text NOT NULL
);

-- S-core. Deployment-wide record that a GPU backend version agreed with CPU on real data (no RLS).
CREATE TABLE cluster_backend_validations (
  id               uuid PRIMARY KEY DEFAULT uuidv7(),
  method           text NOT NULL, backend_versions jsonb NOT NULL,  -- e.g. {"cuvs": "26.10.0", "driver": "..."}
  embedding_field  text NOT NULL, dataset_pin text NOT NULL,
  ari_vs_cpu       real NOT NULL, purity_delta real NOT NULL, graph_recall real NULL,
  passed           boolean NOT NULL,
  validated_at     timestamptz NOT NULL DEFAULT now(), validated_by text NOT NULL
);
```

```sql
-- S-later (lands with merge/split suggestions, N-5). Constraints tied to the object, not to a cluster id.
CREATE TABLE cluster_constraints (
  id uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id uuid NULL REFERENCES projects(id) ON DELETE CASCADE,
  target text NOT NULL, level text NOT NULL,
  kind text NOT NULL CHECK (kind IN ('must_link','cannot_link')),
  item_a text NOT NULL, item_b text NOT NULL, CHECK (item_a < item_b),
  a_bbox_norm real[] NULL, b_bbox_norm real[] NULL,   -- re-attach by IoU after re-detection
  source text NOT NULL CHECK (source IN ('human','import','identity_text','loop')),
  strength text NOT NULL CHECK (strength IN ('hard','soft')),
  confidence real NULL, expires_at timestamptz NULL,  -- machine (soft) constraints expire
  created_at timestamptz NOT NULL DEFAULT now(), created_by text NOT NULL,
  UNIQUE NULLS NOT DISTINCT (project_id, target, level, kind, item_a, item_b)
);
CREATE TABLE cluster_suggestions (
  id uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id uuid NULL REFERENCES projects(id) ON DELETE CASCADE,
  partition_id uuid NOT NULL REFERENCES cluster_partitions(id) ON DELETE CASCADE,
  kind text NOT NULL CHECK (kind IN ('merge','split','move')),
  cluster_uids uuid[] NOT NULL, item_keys text[] NULL,
  score real NOT NULL, evidence jsonb NOT NULL,        -- prototype pairs, RI/RC, judge verdict ids
  status text NOT NULL CHECK (status IN ('open','accepted','rejected','superseded')),
  decided_by text NULL, decided_at timestamptz NULL, revision bigint NOT NULL DEFAULT 1
);

-- S-later (lands with the Cropwright cluster review screens). Judge and human verdicts on a partition.
CREATE TABLE cluster_verification_runs (
  id uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id uuid NULL REFERENCES projects(id) ON DELETE CASCADE,
  partition_id uuid NULL REFERENCES cluster_partitions(id) ON DELETE SET NULL,
  rater_kind text NOT NULL CHECK (rater_kind IN ('human','local_vlm','hosted_import')),
  rater_id text NOT NULL,                 -- 'human:<alias>' or '<model id>@<revision>'
  prompt_version text NULL, sheet_builder_version text NOT NULL, layout text NOT NULL,
  sample_design jsonb NOT NULL,           -- seed, random and targeted counts, controls, salt digest
  blind boolean NOT NULL,
  agreement jsonb NULL,                   -- per kind: raw agreement, kappa, PABAK, flag precision
  job_id uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(), created_by text NOT NULL
);
CREATE TABLE cluster_verdicts (
  run_id uuid NOT NULL REFERENCES cluster_verification_runs(id) ON DELETE CASCADE,
  sheet_alias text NOT NULL,              -- salted alias, never the sheet id
  project_id uuid NULL,
  unit text NOT NULL CHECK (unit IN ('cluster','pair','search')),
  cluster_uid uuid NULL,
  sample_role text NOT NULL CHECK (sample_role IN ('random','targeted','control')),
  verdict text NOT NULL CHECK (verdict IN ('same','different','unsure','invalid')),
  odd_item_keys text[] NOT NULL DEFAULT '{}',
  name text NULL, confidence text NULL CHECK (confidence IN ('high','medium','low')),
  png_sha256 text NOT NULL, manifest_storage_key text NOT NULL, seconds real NULL,
  PRIMARY KEY (run_id, sheet_alias)
);
```

Rules for the tables:

- **Creation order** inside `0001_control_plane`: `cluster_calibrations`, then `cluster_partitions`
  (after `jobs`), `cluster_identities`, `partition_clusters`, `cluster_memberships`,
  `cluster_activations`, `cluster_activation_events`, `cluster_backend_validations`. Every closed list
  (`target`, `level`, `method`, `state`, `transition`) has one Python source in
  `src/domain/clustering_vocab.py` and a test asserts the `CHECK` equals it.
- **RLS.** Every table with `project_id` gets the policy of 02a section 1.4; `cluster_memberships`
  and `partition_clusters` carry `project_id` denormalised (composite FK) so the policy is the
  constant-InitPlan form even under a 1M-row `COPY`, not an `EXISTS` per row.
  `cluster_backend_validations` is deployment data (no RLS, written under `deployment_tx`).
- **Immutability.** A `ready` partition's memberships and clusters never change; a trigger refuses
  `UPDATE` and `DELETE` except by cascade (the `config_revisions_immutable` pattern). Human edits
  create a derived partition (`method = 'edit'`, parent = serving), so every state is reproducible
  and roll-back is a pointer move.
- **Activation** is one project-scoped transaction:
  `UPDATE cluster_activations SET pending_partition_id = :new, revision = revision + 1 WHERE ... AND
  serving_partition_id = :expected AND revision = :rev` (zero rows: 409 `partition_conflict`), plus
  the event row and a `search_projection_tasks` row (D-47 pattern) so the projection cannot be
  forgotten. The projection job flips `serving_partition_id := pending_partition_id` with a fenced
  write at the end.
- **Id carry-over** (in the run, before `ready`): Jaccard matrix of new clusters against the parent's
  clusters over shared members, Hungarian matching (`linear_sum_assignment`) on Jaccard >= 0.5; a
  matched cluster keeps its `cluster_uid` (`survived`); unmatched new ones get a new uid; MONIC
  transitions `split`, `merged`, `absorbed` recorded with `predecessor_uids`. Anchored identities
  are pinned: their members are fixed nodes (`is_membership_fixed`) or contracted to one node, and a
  run that would move an anchored member is refused for auto-activation.
- **Auto-activation policy** (J3): a run activates automatically only when its `diff` shows 0
  anchored or validated moves and id stability >= 0.95 of the parent's clusters; otherwise it stays
  `ready` and waits for a person in Cropwright.
- **Retention:** keep the serving partition, the previous serving one and at most 2 `ready`
  candidates per key; the reaper deletes older `discarded`/superseded partitions in batches of 10,000
  rows after 30 days (02a section 9). Size: about 150 B per membership row with index, about 150 MB
  per 1M-item partition.

### 5.2 Celery job kinds and queues ([02b](program/02b_jobs_locking_events.md) section 2.1 format)

| Kind | Project | Queue | Exclusive group, scope | Auto-resume | Soft / hard | Max wait | Tier | Replaces |
|---|---|---|---|---|---|---|---|---|
| `cluster_run` | yes, or NULL for core faces and images | `cluster` (new) | `cluster_run`, `<target>:<level>:<embedding_field>` | no | 6 h / 6 h 15 m until CL-4 measures the 860k cost | 24 h | S-core | the IVF retrain inside `autolabel`, `region_cluster`, `persons_cluster` (see note) |
| `cluster_projection_sync` | yes or NULL | `jobs` | `cluster_projection`, same scope | yes (idempotent: each item write is fenced by `pending_partition_id`) | 3 h / 3 h 15 m | 24 h | S-core | the write-back stage of today's pass |
| `cluster_suggest` | yes | `cluster` | `cluster_suggest`, same scope | no | 3 h / 3 h 15 m | 24 h | S-later | refine-per-cluster as a split proposal |
| `cluster_verify` | yes | `jobs` (calls the local VLM endpoint) | `cluster_verify`, partition id | no | 3 h / 3 h 15 m | 24 h | S-later | research-only judge runs, for the local VLM only (no product-side hosted judge, owner decision O4) |
| `op.periodic.cluster_refresh` (beat) | per project | `periodic` | advisory lock `CLUSTER_REFRESH` | - | - | - | S-core | today it starts `autolabel`; it starts `cluster_run` instead when the serving partition's `n_items` grew past the profile's growth rule and ingest is quiet |

Note on persons: the program already moves `POST /persons/cluster` to the `persons_cluster` kind
(03b 4C-7). The recommendation is that this kind **is** `cluster_run` with `target = 'face'` and
`project_id` NULL (one wrapper, one record). If 4C-7 has already landed when this is adopted, the
smaller change is that `persons_cluster` writes a `cluster_partitions` row with `method =
'union_find'` and keeps its kind name until the faces research lands.

Queue `cluster`: service `cluster-worker` (api image), prefork, concurrency 1,
`--max-tasks-per-child 1` (a fresh child releases GPU memory, as `gpu_train` does), profile
`curation`, CPU in v0.7.0; `CLUSTER_GPU_ID` only after a passing `cluster_backend_validations` row
and the owner's yes on a RAPIDS image layer (J4). Reason for its own queue: an 860k graph run of
hours must not hold one of the four `jobs` slots that serve request-started work.

Incremental assignment is **not** a job: the embedding stage of ingest assigns each new item to the
serving partition's nearest prototype (prototypes read from Postgres and cached per
`config_epoch`), with the calibrated bands, and writes the OpenSearch fields with
`cluster_assigned_by = 'incremental'`. No band calibration yet means `state = 'assigned'` to the
nearest IVF centroid, which is today's behaviour, recorded as such.

### 5.3 OpenSearch fields

Items index (`{prefix}{slug}__items`) and the fixed keys of a `region_boxes` element
(`src/domain/region_boxes.py`):

| Field | Type | Meaning | Replaces (deleted in the same change) |
|---|---|---|---|
| `cluster_partition_id` | keyword | partition the fields below belong to | - |
| `cluster_uid` | keyword | stable cluster identity (`cluster_identities`) | `cluster_id` (integer), `cluster_subid`, the `RESIDUAL_CLUSTER_ID_OFFSET` namespace and "class clusters" (`cluster_id == class_id`) |
| `cluster_state` | keyword | `assigned`, `review`, `unassigned` | negative `cluster_id` for unassigned |
| `cluster_score` | float | calibrated similarity to the best prototype | `cluster_distance`, `cluster_distance_cluster_id`, `cluster_nearest_id` |
| `cluster_margin` | float | score minus the second-best cluster's | - |
| `cluster_assigned_by` | keyword | `run`, `incremental`, `edit` | - |

`excluded_prior_cluster_id` and `excluded_prior_cluster_subid` are deleted: an item that is
re-included is assigned again incrementally. `FALSE_POSITIVE_REGION_CLUSTER_ID = -100` becomes a
box state, not a cluster id (the region plan owns that field). The class of an item stays
`class_id` (+ the `class_name` projection); a cluster is never a class. The core faces index keeps
`person_id` in this program (02 section 3.2); the faces lane adds `person_state` and
`person_partition_id` with the same meaning when face grouping moves to partitions (S-later).
`tests/test_no_legacy_region_scalars.py` gains a sibling ratchet: no `cluster_subid`,
`RESIDUAL_CLUSTER_ID_OFFSET` or integer `cluster_id` in `src/`.

### 5.4 Config kind

`config_documents.kind` gains `cluster_profile`; `config_activations.axis` gains `clustering_item`
and `clustering_region_box` (core faces and images use a deployment document, `project_id` NULL).
Body (pydantic `ClusterProfile`, `src/config/cluster_profile.py`):

| Key | Type | Meaning |
|---|---|---|
| `target`, `level`, `embedding_field` | enums | which space |
| `method` | enum of `cluster_partitions.method` | `ivf` and `minibatch_kmeans` in v0.7.0 |
| `params` | object per method | e.g. IVF `n_clusters_rule`, train sample cap; Leiden `gamma` or `calibration_id` |
| `pool` | object | the residual gate (confident sources excluded, exclusions, holdout) |
| `assign` | object | `bands: calibration_id or null`, `prototypes_per_cluster` (default 1 until CL-7), `margin` |
| `refresh` | object | growth rule, max deferral, retry backoff (today's `OP_CLUSTER_REFRESH_*` values) |
| `activation` | enum | `auto_when_safe` or `human` |
| `backend` | enum | `cpu` or `gpu_if_validated` |

The env variables whose values move into the profile (`OP_CLUSTER_REFRESH_INTERVAL_S` stays as the
beat interval; `OP_CLUSTER_REFRESH_GROWTH_THRESHOLD`, `OP_CLUSTER_REFRESH_MAX_DEFERRAL_S`,
`OP_CLUSTER_REFRESH_RETRY_BACKOFF_S`, `OP_CLUSTER_REFRESH_AUTO_LABEL`, `OP_IVF_RETRAIN_GROWTH`,
`OP_IVF_RETRAIN_MIN_INTERVAL_S`, `OP_IVF_RETRAIN_CHECK_S`, `OP_MAX_REFINE_MEMBERS`,
`OP_RESIDUAL_EMBEDDING_FIELD`) are retired through `src/config/retired_env.py` (J17).

### 5.5 API and wire (proposed routes, written without a leading slash because they do not exist)

| Route | Method | Body / answer | Tier |
|---|---|---|---|
| `curation/projects/{project}/clusters/runs` | POST | `{target, profile_revision?}` -> 202 `{job_id, partition_id}` | S-core |
| `curation/projects/{project}/clusters/partitions` | GET | list of `ClusterPartition` (status, method, space, counts, metrics, serving flag) | S-core |
| `curation/projects/{project}/clusters/partitions/{partition_id}` | GET | one `ClusterPartition` with `diff` against the serving one | S-core |
| `curation/projects/{project}/clusters/activation` | PUT | `{target, expected_serving_partition_id, expected_revision, partition_id}` -> 202 `{job_id}`; 409 `partition_conflict` | S-core |
| `/curation/projects/{project}/clusters` (exists) | GET | `ClusterCard[]` of the serving partition: `cluster_uid`, `display_no`, size, state counts, prototypes, name, anchored | S-core (shape change) |
| `curation/projects/{project}/clusters/{cluster_uid}` | PATCH | rename or anchor, with `expected_revision` | S-core |
| `curation/projects/{project}/clusters/suggestions` | GET; POST `.../{id}/decision` | merge, split and move proposals with evidence; a decision creates an `edit` partition and, on reject, a cannot-link | S-later |
| `curation/projects/{project}/clusters/{cluster_uid}/verification` | GET | judge and human verdicts beside members | S-later |

Deleted in the S-core change (no parallel routes): `/curation/projects/{project}/regions/cluster`
and `/curation/projects/{project}/regions/cluster/status` (become `clusters/runs` with
`target = region_box` and the job routes); `/curation/projects/{project}/clusters/refine/{cluster_id}`
and `/curation/projects/{project}/regions/clusters/refine/{cluster_id}` (become an `edit` run that
splits one cluster, then S-later split suggestions). Item and box wire objects carry one nested
`cluster: {uid, partition_id, state, score}` through `src/services/curation/wire.py`; `make
contracts` in the same commit.

### 5.6 Who owns each seam

| Seam | Plan file to change | Lane |
|---|---|---|
| Tables 5.1 S-core, retention rows | [02a](program/02a_postgres_schema.md) (new section "Clustering"), section 9 | B2 stage 1, lane 1A (`0001_control_plane`, `src/db/models/clustering.py`) |
| `project_state` keys `last_clustering`, `region_cluster_partition_marker`, `region_cluster_refine_marker`, `cluster_refresh` removed | [02a](program/02a_postgres_schema.md) section 4, [02](program/02_control_plane_database.md) section 3.2 | 1A |
| Queue `cluster`, service `cluster-worker` | [02b](program/02b_jobs_locking_events.md) 7.3, [03a](program/03a_b2_w1_queue_infra.md) 1B-2 | 1B, 1C |
| Kinds `cluster_run`, `cluster_projection_sync`; `region_cluster` removed; IVF retrain out of `autolabel` | [02b](program/02b_jobs_locking_events.md) 2.1, [03b](program/03b_b2_w2_job_families.md) lane 4B (owns `region_cluster_jobs.py`) and 4D (owns `autolabel`) | 4B (new "clustering" family), 4D removes the retrain |
| Persons as `cluster_run(target=face)` | [03b](program/03b_b2_w2_job_families.md) 4C-7 | 4C |
| Beat `cluster_refresh` starts `cluster_run` | [03c](program/03c_b2_w3_beat_reaper_workers.md) | stage 6 |
| OpenSearch fields 5.3, the projection writer, the ratchet test | [02](program/02_control_plane_database.md) 3.4 | 4B (mapping migration through `src/clients/curation_opensearch/` migrations) |
| `cluster_profile` kind and axes | [02a](program/02a_postgres_schema.md) section 4 | B2 stage 2 (config store) |
| Routes and wire 5.5 S-core | [curation_api_contract.md](curation_api_contract.md) | 4B with `make contracts` |
| S-later tables and routes | this plan, section 8 lanes L3-L5 | after a research PASS |
| Loop partition reader on `cluster_partitions` | auto-label research plan (#398) | 4D or its later lane |

## 6. Prototype protocol

The full protocol is in [research_clustering_plan_b_protocol.md](research_clustering_plan_b_protocol.md).
Summary:

- **Where:** tracked `scripts/research/clustering/` with exact pins (`requirements-cpu.txt`,
  `requirements-gpu.txt` and a committed `pip freeze` lock per venv); raw output in
  `artifacts_local/research/clustering/` (gitignored). Tracked because every published number needs a
  reproduce command from a clean checkout ([docs/research/CLAUDE.md](../research/CLAUDE.md)); the
  consolidated note's section-4 numbers rest on untracked scratch scripts and cannot be reproduced,
  which this avoids.
- **Venvs:** two plain Python 3.12 venvs outside the repo (CPU; GPU with RAPIDS 26.10 cu13), never the
  product `.venv`, never Docker builds.
- **Data:** public labelled sets (COCO val/train, Open Images V7, CUB-200, SOP, LFW, CALFW, CPLFW,
  CARS196 local), public real vector sets for timing only, and set P (owner-run; results stay
  private).
- **Bars:** registered in a docs commit before any score (phase R0); verdicts PASS, FAIL,
  INCONCLUSIVE.
- **Judge:** balanced calibration, blind human re-judge with salted aliases, prediction-powered
  purity, faces and plates local only.
- **GPU:** GPUs 0 and 2 only, one job at a time under a shared `flock`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `uptime` recorded, pause when a peer asks.
- **Results:** public numbers in `docs/research/` with backing JSON in `docs/benchmarks/<exp>/`
  (`SHA256SUMS`); set P in `artifacts_local/` only.

## 7. Production design on the new foundation

### 7.1 Flow

```text
ingest -> embed -> incremental assign (serving partition prototypes, calibrated bands)
       -> OpenSearch item fields (cluster_assigned_by = incremental)

beat op.periodic.cluster_refresh (advisory lock CLUSTER_REFRESH, per project)
  -> growth rule of the active cluster_profile met and ingest quiet
  -> start_job cluster_run (queue cluster; unique index = one run per project/target/level/field)

cluster_run (JobTask wrapper: bind project, lease, heartbeat, fenced terminal write)
  1. snapshot the pool through a point-in-time read (sorted by item key; input_digest)
  2. kNN graph (FAISS CPU or, if validated, cuVS GPU), mutual edges
  3. method from the profile (ivf | minibatch_kmeans | later graph_hac | leiden_cpm | ...)
     with anchored members pinned and hard constraints applied
  4. prototypes per cluster; scores and margins; abstention bands from the calibration
  5. id carry-over against the serving partition (Jaccard >= 0.5, Hungarian), MONIC transitions
  6. COPY memberships and clusters, compute label-free metrics and the diff; status ready
  7. auto-activate if the profile allows and the diff is safe; else wait for a person

activation (request or auto) -> OCC on cluster_activations, event row, projection task row
cluster_projection_sync (queue jobs) -> bulk-write only the items whose assignment changed,
  fenced by pending_partition_id -> flip serving_partition_id -> event cluster.activated
```

### 7.2 Properties

- **Multi-user correctness.** Exclusivity is the `jobs` unique index; activation and renames are OCC
  writes; no in-process or file locks; caches (prototypes) are keyed by `config_epoch` and the
  serving partition id and are never a source of truth. About 30 API workers and several worker
  containers may read and write at once.
- **Fail closed.** A run whose embedding model or preprocessing differs from its calibration's
  refuses to use the calibration; a GPU backend without a passing validation row refuses; an
  activation with a stale expected partition is 409; a projection whose fencing token is stale
  writes nothing.
- **Reads during a switch.** Cluster views filter on `cluster_partition_id = serving`; items being
  re-projected drop out of the old view until the flip, and the view shows the projection job's
  progress. At the measured write-back rate (0.31-0.42 ms per crop) a full 1M-item projection is
  about 5-7 minutes; a typical re-run changes far fewer items because only changed rows are written.
  Rejected alternative: blue/green field slots on every item (doubles the fields and every reader).
- **The auto-label loop** reads roots through its `PartitionReader` protocol
  (`src/services/curation/assist_loop/partition.py`). With immutable partitions, the loop's pinned
  member snapshot becomes a reference to `(partition_id, cluster_uid)` plus the lock-rule state; the
  detector-class root partition of #398 is `method = 'detector_class'` (roots keyed by class name).
  The interface the auto-label plan needs from this one: a serving partition per target, immutable
  memberships, `cluster_uid` keys, and a `review` state that never propagates.
- **Human operations** (rename, anchor, merge, split, move) create an `edit` partition derived from
  the serving one, activate it at once (small diff projection) and, for merges and splits accepted
  from suggestions, record must-link or cannot-link constraints tied to item keys and box geometry.
  Machine merges keep machine provenance and are never written as validated (the lock rule in
  `src/clients/occ_locks.py` stays the only class gate; cluster moves never change a class).
- **Embedding change.** A new encoder or preprocessing (embeddings plan, v0.8.0 F9) creates a new
  `embedding_field` or `preprocess_id`, so it gets its own activation key and partitions; the old
  partition keeps serving until the new one is activated. Calibrations never transfer.
- **Faces** keep their data in the core faces index this program; their grouping runs as
  `cluster_run(target='face')` and records partitions, so the research winner replaces the body of
  `similarity_components` and adds stable ids without a new route.
- **Observability.** Each run records stage durations, peak RSS and VRAM, backend detail and the
  label-free metrics; the events table carries `cluster.run.progress` and `cluster.activated`.

## 8. Landing lanes

| Wave | Lane | Files (main) | Red test (fails before, passes after) | Green control |
|---|---|---|---|---|
| v0.7.0 B2 stage 1 | L0 schema | `src/db/models/clustering.py`, `0001_control_plane` | `tests/db/test_schema_matches_plan.py` lists the S-core tables, the partial uniques and the immutability trigger; `tests/db/test_cluster_partition_immutable.py`: updating a `ready` partition's membership raises | upgrade head on an empty database |
| v0.7.0 B2 stage 4 | L1 cluster runs | `src/services/clustering/` (run, carry-over, projection), `src/tasks/clustering.py`, `src/config/job_kinds/clustering.py`, `src/config/cluster_profile.py`; deletes `region_cluster_jobs.py`, `last_run.py`, `refresh_state_store.py`, `cluster_ids.py` offset namespace | `tests/clustering/test_carry_over.py`: the same vectors in permuted order keep every `cluster_uid` and give ARI 1.000 (fails on today's IVF: ARI 0.306); `tests/clustering/test_activation_occ.py`: a stale expected partition gives 409; `tests/clustering/test_projection_fenced.py`: an older projection cannot overwrite a newer partition's fields; `tests/db/test_row_level_security.py` covers the new tables; `tests/test_no_cluster_id_namespace.py` (no `RESIDUAL_CLUSTER_ID_OFFSET`, `cluster_subid`, integer `cluster_id` in `src/`) | an IVF run on a 1,000-item fixture produces a `ready` partition and a served view equal to today's assignment |
| v0.7.0 B2 stage 4 | L1b auto-activation guard | same | `tests/clustering/test_auto_activation_guard.py`: a run that moves an anchored member stays `ready` | a run with id stability 1.0 activates |
| v0.7.0 B2 stage 4 | L1c incremental bands | ingest embed stage | `tests/clustering/test_incremental_bands.py`: an item below the review band is `unassigned` (today every item gets the nearest centroid) | without a calibration, behaviour equals today's |
| v0.7.0 B2 stage 6 | L2 beat | `src/tasks/periodic.py` | `tests/tasks/test_cluster_refresh_starts_cluster_run.py` | no run when growth is below the rule |
| after tag, research | R0 | docs and `scripts/research/clustering/` harness | metric helpers seen failing on a wrong formula (file b section 4); judge calibration guards (file b section 5.1) | existing research tests pass |
| on CL-2 / CL-4 PASS | L3 method | method value, params model, dependency, `ATTRIBUTION.md` | the method's partition on the pinned test fixture reproduces the research score within its interval | IVF path unchanged |
| on CL-5 PASS | L4 suggestions and constraints | S-later tables, routes, Cropwright screens | a rejected merge writes a cannot-link and the next run keeps the two apart | an accepted merge keeps both names |
| on CL-6 PASS | L5 faces | `person_grouping.py` body, persons partition, `person_state` | chaining fixture: two identities bridged by one look-alike stay apart | union-find fixture equality removed with the old body |
| v0.8.0 with Cropwright review | L6 verification | S-later verification tables, `cluster_verify` | a hosted rater kind is refused for face sheets | a local run records verdicts |

Every landing: red/green tests, `make contracts` with any wire change, docs and a CHANGELOG entry,
a live check on a throwaway COCO project (`-p` name), Cropwright pages opened in a browser with
full-page screenshots at desktop and 430 px.

## 9. Risks and stop rules

| Risk | Mitigation |
|---|---|
| S-core grows the B2 scope | it replaces six file and marker mechanisms that B2 must move anyway; J1 offers the record-only minimum |
| Membership tables grow | retention 5.1; COPY writes; per-project size reported in `openprocessor db verify` |
| A held-out half wears out | one read per method family; fresh sets for acceptance (file b 3.2) |
| The B5 crop path changes vectors under the research | parity check of a 1,000-crop sample (cosine >= 0.999) before each window |
| Synthetic wins presented as quality | the scorecard refuses quality fields on synthetic data; every quality number links its scorecard JSON |
| GPU nondeterminism (nn-descent) | GPU builds the graph only; validation row per library version; exact or brute-force for agreement runs |
| Judge bias or drift, unblinded human | balanced calibration, controls, salted aliases, a fresh subagent per case, PPI with the human subset |
| Biometric and plate leakage | local only; refusal by declared kind (OI-2); set P never to a hosted model without a named-sample yes |
| leidenalg licence | J5; scikit-learn fallback |
| Projection write load on OpenSearch during ingest | changed-items-only projection; the projection job yields to ingest through the queue's concurrency 1 |

Stop rules (detail per experiment in file b section 7): a line that fails its registered bar on the
test half of a fresh real set is parked with an evidence comment on #346, the product keeps its
current method, and no further spend goes to it; an INCONCLUSIVE read is repeated once only with a
re-registered, larger set; a method whose partition cannot keep protected moves at 0 and id
stability >= 0.95 does not ship regardless of its quality.

## 10. Owner questions

| ID | Question | Recommendation |
|---|---|---|
| J1 | Build the S-core partition seams (5.1-5.5) in the v0.7.0 rebuild, or only record runs (a `cluster_partitions` row, no memberships, no activation)? | S-core: it replaces six file mechanisms and the `cluster_id`/`class_id` overload, and makes every later method a config change |
| J2 | Membership snapshots in PostgreSQL (about 150 MB per 1M items), or a Parquet blob per partition behind the storage seam? | PostgreSQL: diff, carry-over, rollback and candidate views are queries; size is bounded by retention |
| J3 | Auto-activation when the diff is safe, or every activation by a person? | auto when 0 anchored or validated moves and id stability >= 0.95; else a person |
| J4 | A dedicated `cluster` queue and `cluster-worker`; RAPIDS (cuVS, cuML, cuGraph) only after a GPU validation passes? | yes; CPU in v0.7.0; the RAPIDS layer decided after CL-9 |
| J5 | igraph and leidenalg (GPL) as product dependencies if Leiden wins? | decide only on a Leiden PASS; fallback is scikit-learn average linkage on bounded blocks |
| J6 | If average-linkage HAC wins but needs > 150k items: a block-wise scikit-learn design, or a C++ ParHAC build? | block-wise only; no C++ build in this repo |
| J7 | Set P use: which samples, and may a one-line aggregate verdict per experiment be committed? | the existing 91,184-crop vectors plus a 20,000-crop stratified sample and a face sample; commit nothing from set P except, with your yes, "PASS/FAIL on set P" |
| J8 | Set P sheets to a hosted judge? | never by default; only for a sample you name |
| J9 | Faces on set P (local VLM and you only)? | yes, local only, aggregates private |
| J10 | Publish aggregate face numbers from LFW, CALFW and CPLFW? | yes with a licence caveat; nothing per image |
| J11 | Run the viewpoint line on CO3D and Objectron (non-commercial, local, aggregates only)? | yes, small (CL-7) |
| J12 | Human re-judge tool: CSV beside aliased PNGs, a static HTML form, Label Studio, or Cropwright? | CSV now; Cropwright once S-later review screens exist |
| J13 | Build the Message Batches client (needs an API key you provide)? | only if you provide a key; else the sheet-file subagents |
| J14 | About 8-10 hours of your time over three windows; a second person for a human-human ceiling on 30 sheets? | yes if a delegate exists who has not seen the E0 disagreement lists |
| J15 | Three research GPU windows after the tag on GPU 0 or 2 (about 5, 12 and 6 GPU hours)? | yes, one at a time, announced to peers |
| J16 | Verification tables and the local `cluster_verify` job in the product (v0.8.0, with the Cropwright review screens)? | yes, later; research keeps files until then |
| J17 | Move the cluster refresh and IVF retrain env values into `cluster_profile` and retire the variables? | yes, with the S-core change |

## 11. Required changes to other plan files (for the integration agent)

| File | Change |
|---|---|
| [02_control_plane_database.md](program/02_control_plane_database.md) | 3.2: rows "cluster-refresh growth state" and "region-cluster markers, `last_clustering.json`" go to `cluster_partitions`/`cluster_activations`, not `project_state`; 3.4: the cluster ids on items are the projection of 5.3; 4.2/4.4: `region_cluster` and the IVF retrain become `cluster_run` |
| [02a_postgres_schema.md](program/02a_postgres_schema.md) | add the S-core tables (a "Clustering" section after section 5); `project_state.key` CHECK drops `cluster_refresh`, `region_cluster_refine_marker`, `region_cluster_partition_marker`, `last_clustering`; `config_documents.kind` adds `cluster_profile`; `config_activations.axis` adds `clustering_item`, `clustering_region_box`; section 9 retention rows for partitions and memberships |
| [02b_jobs_locking_events.md](program/02b_jobs_locking_events.md) | 2.1: add `cluster_run`, `cluster_projection_sync`; remove `region_cluster`; `persons_cluster` per 5.2 note; 4.1: `IVF_RETRAIN` advisory lock removed (folded into `cluster_run`); 7.3: queue `cluster`, service `cluster-worker` |
| [03a_b2_w1_queue_infra.md](program/03a_b2_w1_queue_infra.md) | 1A-2 adds `src/db/models/clustering.py`; 1B-2 `TaskQueues` adds `cluster`; 1C-2 adds the `cluster-worker` service |
| [03b_b2_w2_job_families.md](program/03b_b2_w2_job_families.md) | 4B becomes the clustering family (L1, L1b, L1c of section 8) and owns the OpenSearch field migration; 4C-7 per the persons note; 4D removes the IVF retrain from `autolabel` and reads the serving partition |
| [03c_b2_w3_beat_reaper_workers.md](program/03c_b2_w3_beat_reaper_workers.md) | `op.periodic.cluster_refresh` starts `cluster_run`; the reaper prunes partitions per 5.1 |
| [00_index.md](program/00_index.md), [00b_traceability.md](program/00b_traceability.md) | new decision rows for J1-J4 and J17 if approved; #346 and #406 rows point to this plan and file b; the former `issue-346-plan.md` and `issue-406-plan.md` are deleted |
| [12_prior_plans_reconciliation.md](program/12_prior_plans_reconciliation.md) | record the two deleted issue plans as absorbed here |
| [research_gpu_pipeline_plan.md](research_gpu_pipeline_plan.md) | prefix its experiments and windows `GP-` (this plan uses `CL-` and `CRW-`) |
| [cluster_search_makemodel_plan.md](cluster_search_makemodel_plan.md) | sections 2 (principles) and 7 (judge) stay as background; its record-shape and judge-wave content is superseded by this plan and file b |
| [v080_platform_breadth_plan.md](v080_platform_breadth_plan.md) | #348 trigger text: place recall is not a clustering experiment; F9 embedding migration creates a new partition key (7.2) |
| auto-label research plan (#398, parallel lane) | `DetectorClassPartition` records `method = 'detector_class'`; the loop reads `(partition_id, cluster_uid)`; `review` state never propagates |
| near-duplicate research plan (#416, parallel lane) | duplicate groups are must-link input to `cluster_run` and split units in every research split; its thresholds may use `cluster_calibrations` only if the owner wants one table for both |
| embeddings research plan (#347, #365, parallel lane) | every vector bundle carries `embedding_model` and `preprocess_id`; a change creates a new activation key; #435 (frame vector) is decided there |
| plate-text research plan (#401, parallel lane) | identity-text links enter as `cluster_constraints.source = 'identity_text'` (S-later) |
| [docs/design/README.md](README.md) | done in this commit: rows for the two deleted issue plans removed, row for file b added |
