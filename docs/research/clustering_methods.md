# Residual-pool clustering methods

Status: research snapshot (2026-05 to 2026-09); measured on a single workstation GPU and a ~350k-item embedding corpus (plus a 2,469-item ablation corpus); numbers will drift.

Implemented in the public repo: yes, for the methods below. Registry and default (`ivf`): `src/services/curation/clustering/methods/__init__.py`. Methods: `ivf.py`, `ivf_store.py` (persisted centroids), `ahc.py`, `hdbscan.py`. AHC refine: `src/services/curation/clustering/refine.py` (`POST /clusters/refine/{cluster_id}`). Ingest-time assignment: `src/services/curation/clustering/ivf_ingest.py`. Retrain gates: `OP_IVF_RETRAIN_GROWTH`, `OP_IVF_RETRAIN_MIN_INTERVAL_S`, `OP_IVF_RETRAIN_CHECK_S` (see `env.template`). The refine member cap is `OP_MAX_REFINE_MEMBERS` (public default 8000; the numbers below were measured with a cap of 2000). Per-method selection uses the `clustering_method` query parameter on the pipeline start routes. UMAP is retired as a clustering pre-processor in the default path; HDBSCAN and AHC remain selectable.

## 1. The problem

The corpus is about 350k detected crops, each with a 1024-d L2-normalized embedding (cosine objective). Two cohorts exist:

| Cohort | Size | Handling |
|---|---|---|
| Confident (model or human labeled) | ~219k | Label kept; `cluster_id` mirrors the class id; never clustered |
| Residual (low confidence, unmatched, proposals) | ~128k | Clustered into candidate buckets so a human can review in bulk |

Clustering only concerns the residual cohort: group the items the classifier could not label so a human can validate them quickly. It is not identity discovery and not a search index. That framing drove the method choice.

## 2. Methods evaluated

### 2.1 UMAP + AHC (retired)

Reduce 1024-d to 50-d with UMAP, then complete-linkage cosine AHC at `distance_threshold=0.25`. UMAP optimizes local-neighbourhood structure at the cost of global distances, which clustering depends on. At threshold 0.25 it produced a single cluster holding 97.7% of the pool. UMAP is a visualization tool, not a clustering pre-processor.

### 2.2 AHC on raw embeddings (`ahc`)

Build a sparse cosine kNN connectivity graph (k=30; cuML on GPU when available), then sklearn `AgglomerativeClustering` with `linkage='complete'`, `metric='cosine'`, `distance_threshold=0.25`, constrained by that graph. Complete linkage merges only if the maximum pairwise distance stays under the threshold, the strongest guard against bleed-over. Strengths: tight, high-purity clusters and no `n_clusters` knob. Weakness at scale: sklearn's merge loop is one C call that holds the GIL. At n=347k it ran about an hour on one core, peaked near 124 GB RAM, and starved the worker heartbeat until a watchdog killed the job. Keep it for small n and for refine.

### 2.3 HDBSCAN on raw embeddings (`hdbscan`)

cuML GPU HDBSCAN builds a single-linkage tree and picks stable clusters; the only knob is `min_cluster_size`, and noise is labeled `-1`. It needs density gaps. These embeddings are continuously dense (classes vary smoothly with pose, colour and angle). A sweep of `min_cluster_size` in {5, 20, 50, 100, 200} by {eom, leaf} produced either one cluster holding 95-100% of points or 100% noise. The algorithm is sound; this data has no density structure for it. Kept selectable for future curated subsets that do.

### 2.4 FAISS IVF / k-means (`ivf`, default)

Fit K=512 centroids with FAISS (GPU) and assign every embedding to its nearest centroid.

| Requirement | What IVF gives |
|---|---|
| Manageable number of buckets | Fixed K=512 |
| Works on continuously dense data | Partitioning needs no density gaps |
| Every item gets a home | No noise label |
| Balanced buckets | Largest bucket under 1% of the pool; median ~96 items at n=128k |
| Stable across reruns | Same item, same bucket until centroids retrain |
| Speed | GPU k-means ~12 s at 50k, ~30 s at 128k |
| Streaming | Centroids persist; new items assign at ingest in O(1) |

Trade-offs: a Voronoi cell can split a tight visual group at a boundary (refine addresses this); bucket ids are not semantic; retraining moves centroids and can change an item's bucket, so retraining is infrequent and a cheap `reassign_only` path re-sorts against current centroids.

IVF partitioning must not be coupled to a search index even though both use k-means centroids: pre-partitioning through a search structure would reintroduce global distortion.

## 3. Coarse-to-fine: IVF then AHC refine

IVF gives 512 buckets (a few hundred items each). When a reviewer sees a mixed bucket, AHC refine (complete linkage, cosine, `distance_threshold=0.25`, capped at a maximum member count) splits it into sub-ids. AHC's single-thread, memory-heavy weakness is irrelevant at a few thousand items, and its threshold-controlled tight splits are what "split this bucket" needs. This is a hierarchy, not an ensemble.

## 4. Streaming architecture

- **Bounded-memory training:** k-means trains on a random sample of at most 50k items (`IVF_MAX_TRAIN_SAMPLE`); FAISS recommends about 256 points per centroid, and 50k is far above 256 x 512. Training transient stays near 200 MB regardless of pool size; the full pool is still assigned.
- **Persist and assign on ingest:** centroids persist in a store shared by the worker and the API. Every residual item gets its nearest centroid at ingest (O(1), process-cached keyed on the centroid file mtime, so a retrain propagates without a restart). With no centroids yet, `cluster_id` stays null.
- **Periodic reassign (`reassign_only=true`):** streams the pool one scroll page at a time (peak RAM about one page, ~8 MB).
- **Full retrain:** refit on a fresh sample, then reassign.

Auto-retrain on an idle worker requires both gates: growth above 1.5x the count at last train, and at least 24 hours since the last train. The count query runs every 30 minutes. The first train needs no cooldown. Operators can always force a retrain.

## 5. Residual-pool gating

An item enters the residual pool only if it has an embedding and is not human-validated, not from a confident source (model, VLM or human), and not excluded. A regression is worth remembering: gating on validated status alone pulled the entire corpus (347k) into the pool and overwrote 219k confident items' `cluster_id`; adding the confident-source gate fixed it and a normalize pass restored 132k clobbered ids.

Items that are blurry, unidentifiable or partial are removed from training and clustering with a reversible exclude flag; every read path filters it, and it is non-destructive.

## 6. Fast embedding fetch

A sequential scroll of ~350k x 1024-d embeddings took about 10 minutes. A frozen Point-In-Time view with 8 concurrent slice readers (paged by `search_after` on `_shard_doc`) is about 8x faster, with a scroll fallback when PIT is unavailable.

## 7. Overlays are additive, not alternatives

Review-sort orderings, scoring (see [curation_scores.md](curation_scores.md)), diverse selection and a 2-d visualization projection never write `cluster_id`; they decorate or reorder an existing assignment. They are exposed through the `GET /methods` capability endpoint and move through `disabled`, `shadow`, `experimental`, `stable`. UMAP for visualization is consistent with section 2.1: it never decides membership, it only makes an existing assignment easier to inspect.

## 8. Phase-2 ablation (HDBSCAN vs AHC vs OPTICS)

Date: 2026-05. The residual pool was empty at run time, so the ablation ran on all 2,469 items that had two embeddings. This biases coherence downward (validated and already-clustered items are mixed in), but the ranking between methods is robust.

Method: L2-normalize; UMAP to 50-d (`metric='cosine'`, `random_state=42`, `min_dist=0.0`, `n_neighbors=15`, or 30 in H5/H6); then cluster with one of: HDBSCAN (sklearn, euclidean, `eom`), AHC (ward with automatic cluster count via a silhouette sweep over 6, 8, 10, 12, 15, 20, 25; or cosine average/complete with a fixed threshold), or OPTICS (sklearn defaults, `min_samples=5`, `max_eps=0.5`). Scores: cosine silhouette, Davies-Bouldin, Calinski-Harabasz (non-noise points), VLM-label coherence (fraction of a cluster's labelled members matching the modal label, averaged), noise fraction, and an operator-throughput proxy (OTP) = mean cluster size x coherence x (1 - noise fraction).

| id | method | embedding | min cluster size | extras | clusters | noise | mean size | coherence | OTP | silhouette |
|----|--------|-----------|-----|---------------------|---:|-----:|------:|-----:|-------:|------:|
| A1 | ahc | embedding A | - | ward, auto-n=15 | 15 | 0.00 | 164.6 | 0.35 | 57.15 | 0.755 |
| H5 | hdbscan | embedding A | 12 | n_neighbors=30 | 79 | 0.05 | 29.6 | 0.22 | 6.15 | 0.812 |
| H6 | hdbscan | embedding A | 12 | n_neighbors=30, min_dist=0.1 | 77 | 0.07 | 30.0 | 0.19 | 5.45 | 0.787 |
| H4 | hdbscan | embedding A | 12 | former default | 90 | 0.05 | 26.0 | 0.18 | 4.39 | 0.842 |
| H2 | hdbscan | embedding B | 12 | | 88 | 0.10 | 25.2 | 0.18 | 4.09 | 0.813 |
| H1 | hdbscan | embedding B | 8 | | 124 | 0.08 | 18.4 | 0.13 | 2.18 | 0.839 |
| H3 | hdbscan | embedding A | 8 | | 127 | 0.07 | 18.0 | 0.12 | 1.93 | 0.854 |
| O1 | optics | embedding A | - | min_samples=5 | 207 | 0.23 | 9.2 | 0.09 | 0.64 | 0.873 |
| A2 | ahc | embedding A | - | average, thr=0.35 | 1 | 0.00 | 2469.0 | 0.14 | 352.71 | n/a |
| A3 | ahc | embedding A | - | complete, thr=0.40 | 1 | 0.00 | 2469.0 | 0.14 | 352.71 | n/a |
| A4 | ahc | embedding B | - | average, thr=0.35 | 1 | 0.00 | 2469.0 | 0.14 | 352.71 | n/a |

(Embedding A is the 1024-d general-purpose encoder embedding; embedding B is the detector-derived embedding.)

A2-A4 are degenerate: the fixed thresholds collapse everything into one cluster, which the OTP proxy scores misleadingly high while silhouette is undefined; they are excluded from the winner comparison. A future sweep over `distance_threshold` (for example 0.10-0.30 in 0.025 steps) is untested.

Two-knob decomposition within HDBSCAN:

| Step | OTP | Delta from H1 |
|---|---:|---:|
| H1 (B, mcs=8) | 2.18 | - |
| H2 (B, mcs=12) | 4.09 | +1.91 (+88%, min cluster size alone) |
| H3 (A, mcs=8) | 1.93 | -0.25 (-11%, embedding alone) |
| H4 (A, mcs=12) | 4.39 | +2.21 (+101%, both) |

About 96% of the H1-to-H4 gain comes from raising `min_cluster_size` from 8 to 12; switching embeddings is roughly neutral. Raising UMAP `n_neighbors` to 30 adds another ~40% (H5).

Best of each family:

| Family | Best | OTP | silhouette | clusters | mean size | coherence |
|---|---|---:|---:|---:|---:|---:|
| HDBSCAN | H5 | 6.15 | 0.812 | 79 | 29.6 | 0.22 |
| AHC | A1 | 57.15 | 0.755 | 15 | 164.6 | 0.35 |
| Delta | | +829% | -7% | | | +59% |

AHC won on this small corpus because it has no noise bucket (HDBSCAN shed 5-10% to `-1`), it picks a coarser granularity matching the real class diversity (the VLM produced about 20 distinct labels), and per-cluster label agreement nearly doubled. The silhouette dip is the price of larger clusters. The ablation winner was ward AHC with automatic cluster count on embedding A with UMAP `n_neighbors=15`, +52.76 OTP (+1,202%) over H4.

Caveats: the small, non-residual corpus understates absolute coherence; cosine average/complete AHC was untested because the fixed thresholds collapsed; OPTICS was inferior on every axis. Note that this small-n ablation preceded the full-scale (n in the hundreds of thousands) findings in sections 2.1-2.4, where UMAP-based pipelines collapsed and AHC did not scale; the production default is therefore IVF with AHC reserved for refine.

## 9. Decision summary

| Task | Method | Why |
|---|---|---|
| Residual pool, batch | IVF | Continuously dense data wants partitioning; fixed K, balanced, fast, streaming-ready |
| Residual pool, per item at ingest | IVF with persisted centroids | O(1) nearest centroid |
| Split a mixed bucket | AHC refine | Tight threshold-controlled splits at small n |
| Curated subset with density gaps | HDBSCAN | Right tool only if density structure exists |
| Dimensionality reduction before clustering | none | UMAP destroys the geometry clustering needs |

Rule of thumb: partition (IVF) to hand a human browsable buckets; refine (AHC) to split a bucket they flag; never reduce (UMAP) or density-cluster (HDBSCAN) embeddings that have no density gaps.
