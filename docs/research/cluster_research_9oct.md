# Research: Continuous, recursive, self-refining clustering for OpenProcessor

Yes. What you're describing is an established and active area of machine-learning research. More importantly, it addresses a real limitation of conventional clustering: the first clustering pass is rarely the best possible organization of a large, diverse image dataset.

The research overlaps several fields:

- Hierarchical clustering: discovering groups and subgroups at multiple levels.
- Split-and-merge clustering: dividing overly broad clusters and reuniting incorrectly separated groups.
- Iterative clustering refinement: repeatedly improving assignments based on the existing cluster structure.
- Graph community detection: discovering clusters from relationships between items, rather than relying only on centroids.
- Self-supervised deep clustering: using cluster assignments as training signals to improve the embeddings themselves.
- Incremental and dynamic clustering: updating groups as new images arrive without repeatedly reclustering the entire collection.

The distinction that matters most for OpenProcessor is this:

You don't simply want to cluster each of your 500 existing clusters again. You want the algorithm to reconsider the boundaries between those clusters while preserving meaningful distinctions.

That requires allowing items to move between clusters, splitting mixed clusters, merging related clusters, and leaving ambiguous items unassigned.

## 1. What I found in OpenProcessor

I located your public [OpenProcessor GitHub repository](https://github.com/davidamacey/OpenProcessor?utm_source=chatgpt.com) and reviewed its current architecture and curation documentation.

Your project already has useful foundations for this research:

| Existing capability                         | Relevance                                                         |
| ------------------------------------------- | ----------------------------------------------------------------- |
| FAISS/IVF clustering                        | Fast initial grouping of embedding vectors                        |
| Agglomerative hierarchical clustering (AHC) | Refining an individual cluster into smaller groups                |
| OpenSearch vector indexing                  | Finding nearest neighbors across existing cluster boundaries      |
| PE-Core 1024-dimensional embeddings         | Representing visual similarity between item crops                 |
| `cluster_id` and `cluster_subid`            | Storing a two-level cluster hierarchy                             |
| Periodic cluster-refresh worker             | A natural integration point for continuous refinement             |
| Human labels, VLM labels, and locking       | Providing trusted signals and constraints for cluster corrections |

Your current API even exposes `POST /curation/projects/{project}/clusters/refine/{cluster_id}` for AHC refinement of a particular cluster. It also provides a residual-pool UMAP rebuild route.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+2



This reveals an important architectural opportunity: OpenProcessor already supports refining downward inside a cluster. The major improvement would be adding cross-cluster reassignment and controlled splitting/merging across the entire dataset.

One qualification: I could inspect the public repository's architecture, documented interfaces, and data contracts, but not verify every implementation branch from a local checkout. I'll distinguish documented current behavior from recommended new functionality.

## 2. Why reclustering your existing 500 clusters isn't enough

Consider a simplified image dataset:

Cluster 12

[Top 20 Best-Selling Vehicles of 2014, Quarter One - Motor Trend](https://images.openai.com/static-rsc-4/MzuEC7PeQR7YfRGTyLkf2uQFvVEGNILYd2IB9lFEOEI9yYyyPUd9pdDeW07MrsFGjIuF1ZqgffiBTAMCxgxdYpo_494NO55iAzvqrOwgsUJRU2B2Ou01nRe3OTxtli5jP18cUPS45zTUUpbJ5zMZEHYjiuOtv5Fnw3wFSzCzekE?purpose=inline)

Cars + trucks

Cluster 34

[10 Cars That Have Grown an EPA Class Size (or More) Over the Years](https://images.openai.com/static-rsc-4/zzGnmGg1vP4WsABAMj5kajtSXqNAR6Ed91Vj3QC3jU1aKNxCOfzQmxdOJQj80LneNoO_UziS2aqzXRpjCC-Fv2lZ7H5ou_PX7BEaT0fEpnPny_n-wB8X-H9FoqgcBrYvPpd4ApujkLiNc9pQ3zjRakbspIoYM3bc9HmiNjRBqT0?purpose=inline)

Mostly cars

Cluster 78

[The $52,000 Question: F-150 vs Silverado 1500 vs Ram 1500 vs Tundra - Autoblog](https://images.openai.com/static-rsc-4/OjZiV7BsW0ftbzoIbEXH2g-gfgBjW9uwdfEwEBBJWqhRM2mRBdzCB7mpkUZhIX2Z5gbuIKc_hGiuAawV65Bznk82lA-q_tIDcywBrv5uS_HorJCBFGou62hdMGuiFviE6QdDeyNBWNw_9ZJQFx8dBaVo3xDPHQtSUN-Y-s0WtVA?purpose=inline)

Mostly trucks

Global boundary refinement

Refined A: Cars

Members from 12 + 34

Refined B: Trucks

Members from 12 + 78

Illustrative example: refinement can change membership across the original boundaries.

If you perform AHC independently inside clusters 12, 34 and 78, cluster 12 might split correctly into cars and trucks.

But it cannot discover that the cars from cluster 12 should join the cars from cluster 34.

The missing operation is cross-cluster reassignment.

This is where nearest-neighbor graphs and community-detection algorithms become particularly useful.

## 3. Academic methods worth investigating

I found several directly relevant families of algorithms. They solve related problems, but they do not all provide the same functionality.

| Algorithm                  | What it contributes                                                             | Fit for OpenProcessor         |
| -------------------------- | ------------------------------------------------------------------------------- | ----------------------------- |
| Leiden                     | Iterative graph partitioning, moving members between communities                | Excellent                     |
| CHAMELEON                  | Subclusters followed by intelligent merging based on connectivity and closeness | Excellent conceptually        |
| FINCH                      | Fast, automatically generated clustering hierarchy                              | Very good                     |
| HDBSCAN leaf clustering    | Finds finer clusters while retaining outliers                                   | Very good                     |
| Consensus clustering / ECG | Repeated clustering to identify stable relationships                            | Very good                     |
| DeepCluster / SCAN         | Improves learned features using cluster relationships                           | Future research               |
| HIT-Leiden (2026)          | Incrementally updates hierarchical graph communities                            | Highly relevant, experimental |

### A. Leiden: my first choice

[From Louvain to Leiden: Guaranteeing Well-Connected Communities (2019)](https://www.nature.com/articles/s41598-019-41695-z?utm_source=chatgpt.com)

This is one of the strongest candidates for what you're describing.

Leiden starts with a graph in which each item is a node and connections represent relationships between items. It repeatedly optimizes how nodes are assigned to communities, refines the resulting communities and aggregates them.

The crucial difference from independently reclustering your existing groups is that the initial cluster assignment doesn't need to remain fixed.

A node currently belonging to cluster 12 can be reassigned to cluster 34 when the graph provides stronger evidence for that grouping.

Repeated Leiden optimization has mathematical connectivity guarantees for communities in the supplied graph. It does not, however, guarantee that the graph itself encodes correct semantic similarity.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://doi.org\&sz=32)

Scientific Reports+1



For OpenProcessor, I would particularly investigate Leiden with the Constant Potts Model (CPM) objective, weighted nearest-neighbor edges and multiple resolution levels.

CPM has a tunable resolution parameter controlling community granularity. Increasing the resolution generally favors smaller communities.

The parameter is not a universal similarity cutoff; you'll need to calibrate it for your embedding distributions.

### B. CHAMELEON: conceptually very close to your idea

[CHAMELEON: Hierarchical Clustering Using Dynamic Modeling (1999)](https://experts.umn.edu/en/publications/chameleon-hierarchical-clustering-using-dynamic-modeling/?utm_source=chatgpt.com)

CHAMELEON is especially interesting because it explicitly starts with smaller partitions and then joins groups according to two properties:

Relative interconnectivity: How strongly are two clusters connected compared with their internal connectivity?

Relative closeness: How close are those clusters compared with the similarity patterns inside each one?

For example, two groups of vehicles may have moderately high cosine similarity between their centroids. But if those groups have very few strong item-to-item connections, merging them could be a mistake.

CHAMELEON uses evidence about the internal structure instead of centroid distance alone.

It's older research, but its design remains extremely relevant. I would borrow its ideas for an explicit split/merge validation stage even if you use Leiden as your main algorithm.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://experts.umn.edu\&sz=32)

Experts\@Minnesota



### C. FINCH: automatically discovering a hierarchy

[Efficient Parameter-Free Clustering Using First Neighbor Relations (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Sarfraz_Efficient_Parameter-Free_Clustering_Using_First_Neighbor_Relations_CVPR_2019_paper.html?utm_source=chatgpt.com)

FINCH uses first-neighbor relationships to construct successive clustering partitions without requiring you to specify the original number of clusters.

For example, a dataset might have a fine partition, an intermediate partition and a coarse partition. You can inspect each level rather than accepting a single partition.

This makes FINCH attractive for a hierarchy of image collections, particularly for quickly testing whether your 500-cluster representation is too coarse or too fine.

The limitation is that the first-neighbor relationships can propagate undesirable groupings. It still needs validation before accepting a coarse level, especially for heterogeneous imagery.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

CVF Open Access



### D. HDBSCAN: use more of its existing hierarchy

HDBSCAN is already hierarchical internally. Its default `eom` selection extracts persistent clusters, while `leaf` selection generally produces smaller, more homogeneous clusters.

That means you can get further refinement without inventing a new algorithm.

However, repeatedly running HDBSCAN inside each existing cluster is still constrained by those original boundaries. It cannot correct cross-cluster errors unless the items are reconsidered together.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1



I would retain HDBSCAN as a comparison method and potentially use leaf clustering on mixed communities discovered by Leiden.

## 4. The especially relevant 2026 research: HIT-Leiden

2026 research

Hierarchical Incremental Tree Leiden

[Maintaining Leiden Communities in Large Dynamic Graphs](https://arxiv.org/abs/2601.08554?utm_source=chatgpt.com)

This research addresses a major issue with conventional graph clustering: what happens when the graph is continuously changing?

Instead of restarting clustering every time new nodes and connections arrive, HIT-Leiden maintains a hierarchical structure and updates the parts affected by a graph change.

The authors report substantial runtime improvements relative to full recomputation and earlier incremental methods. Those are research benchmark results, not verified performance figures for OpenProcessor.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



I also located a [Rust implementation of HIT-Leiden](https://github.com/randomvariable/hit-leiden?utm_source=chatgpt.com). It's an experimental project with preliminary benchmarks and explicitly recommends independent validation before production use.

This is worth watching if you eventually want a lightweight Rust clustering service running independently of Triton.

For the first implementation, though, I would use mature Leiden libraries and a controlled periodic update process before introducing an experimental dynamic implementation.

## 5. What I would build in OpenProcessor

I'd introduce a Global Cluster Refinement Engine, separate from the inference path.

Image / video ingestion

Triton inference · PE-Core embeddings · OpenSearch

Initial clustering

Existing FAISS / AHC / residual clustering

\~500 clusters

New Global Refinement Engine

Cross-cluster kNN

Mutual-neighbor graph

Leiden partitioning

Split / merge validation

Stability + quality + constraint checks

Versioned cluster hierarchy

Parents · children · assignments · confidence · lineage

New data triggers local updates and periodic global validation

The engine would perform six operations:

1. Build a cross-cluster kNN graph. Search each item's nearest neighbors globally, not only within its existing cluster. Use the original normalized embedding vectors rather than 2D UMAP coordinates.
2. Create robust connections. Give more weight to mutually confirmed nearest-neighbor relationships and consistently strong edges. Avoid letting a few weak bridges connect otherwise unrelated groups.
3. Run Leiden at several resolutions. Initialize from the existing clusters for comparison, but permit reassignment. Evaluate partitions instead of forcing a particular number of clusters.
4. Perform controlled splits and merges. Split communities with evidence of distinct subgroups; merge strongly related groups only when internal consistency and cross-group connectivity justify it.
5. Evaluate the proposed partition. Reject changes that worsen cluster quality, violate trusted labels or create large, poorly supported groups. Keep ambiguous items in a noise or review state.
6. Publish a versioned result. Preserve old assignments, maintain stable cluster identities where possible and record exactly which items moved, split or merged.

### Preventing everything from becoming one giant cluster

This is central to your question. You cannot assume that repeatedly clustering will automatically improve the result.

A bad graph can contain connecting chains that eventually join many otherwise distinct groups. Likewise, repeatedly dragging embeddings toward their current cluster centroids can reinforce incorrect assignments.

I'd implement several independent safeguards.

| Safeguard                        | Why it matters                                                              |
| -------------------------------- | --------------------------------------------------------------------------- |
| Mutual kNN connectivity          | Removes some one-way, weakly supported relationships                        |
| Adaptive similarity requirements | Avoids joining items merely because they are each other's nearest neighbors |
| Leiden CPM resolution            | Controls how readily communities consolidate                                |
| Intra-cluster dispersion checks  | Detects groups with excessive internal variation                            |
| Trusted cannot-link constraints  | Prevents known-incompatible items being placed together                     |
| Stability across repeated runs   | Distinguishes persistent groupings from algorithmic instability             |
| Minimum assignment confidence    | Leaves uncertain items unassigned                                           |
| Merge quality gate               | Rejects changes whose quality evidence is insufficient                      |

No single safeguard guarantees semantic correctness. Together, they make large erroneous merges less likely.

I would not enforce a fixed number of 500 clusters. The correct count might be 320, 740 or something else. The goal is higher-quality separation, not maintaining a particular number.

## 6. Make the hierarchy more expressive

Your existing `cluster_id` and `cluster_subid` are a useful two-level starting point, but I'd move the persistent hierarchy into separate cluster records.

Vehicle — broad semantic category

Cars

Parent cluster A

Trucks

Parent cluster B

Sedans

Subcluster A1

Pickup trucks

Subcluster B1

Illustrative semantic hierarchy, not labels inferred by Leiden itself.

I'd persist the following information separately from the current assignment:

| Field                     | Purpose                                       |
| ------------------------- | --------------------------------------------- |
| `cluster_uuid`            | Stable cluster identity                       |
| `parent_uuid`             | Supports hierarchical grouping                |
| `level`                   | Coarse, intermediate or fine                  |
| `cluster_run_id`          | Reproducible partition version                |
| `membership_confidence`   | How strongly an item belongs                  |
| `embedding_model_version` | Prevents mixing incompatible embedding spaces |
| `cluster_lineage`         | Records split, merge and reassignment history |
| `label_provenance`        | Distinguishes human, VLM and inferred labels  |

A hierarchy represents parent-child relationships. Other attributes—color, camera source, date, viewpoint or location—might be better stored as filters rather than forcing them into exclusive branches of the hierarchy.

This distinction matters when your clusters represent different visual concepts rather than mutually exclusive object identities.

## 7. GitHub projects worth evaluating

These are the repositories I would prioritize. They range from mature production libraries to interesting research implementations.

| Repository                                                                                 | Why I'd look at it                                                           | Priority |
| ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- | -------- |
| [vtraag/leidenalg](https://github.com/vtraag/leidenalg?utm_source=chatgpt.com)             | Flexible Python/C++ Leiden implementation, including CPM and weighted graphs | Highest  |
| [RAPIDS cuGraph](https://github.com/rapidsai/cugraph?utm_source=chatgpt.com)               | GPU graph community detection, Leiden and ensemble clustering                | Highest  |
| [FINCH-Clustering](https://github.com/ssarfraz/FINCH-Clustering?utm_source=chatgpt.com)    | Builds hierarchical partitions from first-neighbor relationships             | High     |
| [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan?utm_source=chatgpt.com)          | Stable/leaf extraction and condensed-tree analysis                           | High     |
| [hit-leiden](https://github.com/randomvariable/hit-leiden?utm_source=chatgpt.com)          | Experimental Rust incremental hierarchical clustering                        | Research |
| [Meta DeepCluster](https://github.com/facebookresearch/deepcluster?utm_source=chatgpt.com) | Deep clustering and representation-learning research; archived repository    | Research |
| [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification?utm_source=chatgpt.com) | Learning image clusters using neighbor consistency                           | Research |

One additional project is [GraphWeave](https://github.com/nevil-mathew/GraphWeave?utm_source=chatgpt.com). It explicitly combines mutual nearest-neighbor graphs, consensus Leiden clustering and iterative centroid refinement. Its reported results are project-provided benchmarks rather than independent evidence, but it is a useful implementation reference for exactly the kind of repeated graph-refinement loop you're contemplating.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



RAPIDS is especially interesting given your NVIDIA inference stack. cuGraph provides GPU Leiden and Ensemble Clustering for Graphs (ECG), while cuML provides GPU HDBSCAN. These are different execution choices for different algorithms; they aren't one interchangeable clusterer.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.rapids.ai\&sz=32)

cugraph-docs 26.06.00 documentation+1



## 8. Should you actually retrain the embeddings?

There are two different interpretations of self-supervised recursive clustering.

| Approach                       | Changes cluster assignments? | Changes neural-network weights? |
| ------------------------------ | ---------------------------- | ------------------------------- |
| Iterative Leiden / split-merge | Yes                          | No                              |
| FINCH / HDBSCAN hierarchy      | Yes                          | No                              |
| Consensus clustering           | Yes                          | No                              |
| DeepCluster / SwAV / SCAN      | Yes                          | Yes, during training            |

With your existing PE-Core embeddings, I would initially avoid retraining the embedding model.

You can achieve considerable improvements by changing how the embeddings are organized without introducing model training, retraining costs or new model-version dependencies.

DeepCluster and SwAV are important research references because they alternate or combine feature learning with clustering-based objectives. SCAN also uses nearest-neighbor consistency to learn semantic categories.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+2



But there is a substantial danger of confirmation bias: if the initial clusters are wrong and you use them as pseudo-labels for model training, the model can become more confident about those errors.

For a future experimental feature, I'd consider a separate embedding-adapter training pipeline that only uses stable clusters, trusted labels and independent validation.

The first production clustering improvement does not need that complexity.

## 9. Continuous refinement at scale

I would not perform a full global reclustering every time a new image arrives.

Instead, your periodic cluster-refresh worker can evolve into a tiered system.

| Frequency               | Operation                                                       | Scope                                      |
| ----------------------- | --------------------------------------------------------------- | ------------------------------------------ |
| On ingestion            | Assign likely cluster using kNN and confidence gating           | Individual item                            |
| Small accumulated batch | Reevaluate uncertain assignments and their neighboring clusters | Local neighborhood                         |
| Scheduled refresh       | Recompute affected graph communities and split/merge candidates | Selected clusters                          |
| Periodic full audit     | Reevaluate global relationships and long-term drift             | Entire dataset or representative hierarchy |

At very large scale, this becomes a microcluster hierarchy.

You can maintain compact microclusters with representative vectors, counts and dispersion statistics, then use those structures to select which larger groups need examination.

This avoids materializing an enormous, unrestricted item-to-item graph for every update.

For streaming background research, BIRCH is a classic online clustering technique, while DenStream models evolving density-based microclusters and outliers. Neither is necessarily a drop-in solution for your high-dimensional visual embeddings, but both provide useful architecture ideas.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://scikit-learn.org\&sz=32)

scikit-learn 1.9.1 documentation+2



I'd also retain a full-quality audit path because local updates can gradually miss relationships that form across previously distant groups.

## 10. Testing whether the new clusters are actually better

This is where I would spend significant effort before allowing automated changes to affect dataset labeling.

Your current API already distinguishes geometric purity from label purity. That's valuable, but neither is sufficient as a standalone measure of semantic correctness.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



I recommend creating a small, curated evaluation collection that deliberately includes difficult cases: visually similar but distinct objects, different views of the same object type, low-resolution images, mixed clusters, rare classes and duplicate or near-duplicate items.

Then compare your baseline with these experiments:

| Experiment                      | What it tests                                 |
| ------------------------------- | --------------------------------------------- |
| Existing FAISS/AHC              | Current baseline                              |
| HDBSCAN leaf refinement         | Finer within-cluster partitioning             |
| FINCH                           | Automatically discovered hierarchy            |
| Global Leiden                   | Cross-cluster reassignment                    |
| Leiden + split/merge validation | Protection against incorrect consolidation    |
| Consensus Leiden                | Stability across runs and graph perturbations |

Measure pairwise precision and recall on independently labeled examples, false-merge rate, correct split recovery, unassigned fraction and assignment stability. Also measure runtime, memory and how many previously labeled items would have changed.

I would prioritize false merges over forcing every image into a cluster. For dataset curation, assigning an uncertain item to review is often preferable to automatically propagating a wrong class label.

## 11. Specific changes I'd recommend for the repository

The current architecture places clustering under `src/services/curation/clustering/`, exposes it through curation routers and runs periodic refresh work from `scripts/curation/`. Those are the appropriate existing integration points.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



Rather than replacing the existing clustering system, I'd build an independent refinement service alongside it.

```

src/services/curation/clustering/
    ├── existing clustering modules
    └── refinement/                  # Proposed
        ├── neighbor_graph.py
        ├── leiden_partition.py
        ├── split_merge.py
        ├── cluster_quality.py
        ├── constraints.py
        ├── hierarchy.py
        ├── incremental_update.py
        └── partition_versions.py

scripts/curation/
    └── cluster_refinement_worker.py # Proposed

```

Its central workflow could look like this:

```
def refine_clusters(project, current_partition):    # 1. Load compatible vectors and trusted constraints.    vectors, metadata = load_embeddings(project)    # 2. Search beyond current cluster boundaries.    neighbors = global_knn(vectors)    # 3. Keep well-supported similarity relationships.    graph = build_mutual_knn_graph(        neighbors, metadata    )    # 4. Produce candidate partitions at different    #    resolutions, allowing members to change groups.    candidates = leiden_multiresolution(        graph, initial=current_partition    )    # 5. Refine mixed groups and evaluate proposed merges.    proposals = propose_splits_merges(candidates)    # 6. Evaluate independently before publishing.    accepted = validate_changes(        proposals, metadata, current_partition    )    return create_versioned_partition(accepted)
```

This is architectural pseudocode, not code intended to run unchanged against the current repository. The actual implementation must reuse OpenProcessor's project-scoped OpenSearch access, optimistic-concurrency handling and human-label locking behavior.

I'd also keep the clustering worker off the critical Triton inference path. Triton should produce embeddings efficiently; graph construction and optimization should be asynchronous downstream operations.

One important consideration: don't use mixed MobileCLIP, ArcFace and PE-Core vectors in one similarity graph. Those represent different embedding spaces. Build model- and task-specific graphs unless you've explicitly learned and validated a compatible shared space.

## 12. My recommended development sequence

Phase 1 — Global kNN + Leiden prototype

Keep the original 500 clusters intact. Build a mutual-kNN graph using PE-Core embeddings and compare several Leiden resolutions. Measure whether misplaced items are reassigned correctly.

Phase 2 — Constrained split/merge and cluster lineage

Add quality gates, stable IDs, partition revisions and parent-child relationships. Allow human-approved changes without accidentally overwriting trusted labels.

Phase 3 — Continuous local refinement

Integrate the new graph operations into the background refresh worker. Update affected neighborhoods as embeddings arrive and run larger scheduled audits.

Phase 4 — GPU scaling and experimental algorithms

Benchmark cuGraph Leiden, ECG and GPU HDBSCAN. Evaluate FINCH and HIT-Leiden as alternative hierarchical or incremental implementations. Consider self-supervised feature adaptation only if clustering improvements plateau.

### Where I would start

I would choose FAISS/OpenSearch for neighbor retrieval, weighted mutual-kNN graphs for relationships, Leiden for global partitioning, and CHAMELEON-inspired rules for conservative split/merge decisions.

That is more suitable to the problem you've described than continually reclustering inside each of your 500 original groups.

UMAP wouldn't be in the critical clustering path initially. Its own documentation cautions that dimensionality reduction can distort density and create artificial separations; your disappointing results with it aren't surprising in that context. I would reserve 2D UMAP mainly for visualization and compare any higher-dimensional use against clustering directly on your original embeddings.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://umap-learn.readthedocs.io\&sz=32)

umap 0.5.8 documentation



The most important research direction for OpenProcessor is the intersection of Leiden, CHAMELEON, consensus clustering and the new incremental Leiden work. Together, those provide a credible route toward a system that continuously reorganizes its clusters, finds missed relationships, preserves meaningful subgroups and becomes more consistent as additional media arrives—without needing to retrain your vision models after every iteration.