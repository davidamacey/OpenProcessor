# v0.6.0 vector store evaluation plan

Status: plan (nothing implemented). Issue: #221 (work package WP-V of
[v060_wave_plan.md](v060_wave_plan.md)); its OpenSearch-tuning arm is #56. Measurement only: no
datastore change ships from this plan without an explicit owner decision.

Question: for the curation embedding index, is OpenSearch k-NN (as shipped, or tuned) the right
engine, or would a dedicated vector store (Milvus, Qdrant, pgvector, LanceDB) be worth a second
service? Answer it with one fair benchmark on the same real embeddings, and with an honest account
of what the datastore does besides vectors.

File:line references were checked against `origin/main` at `877f62c9`; re-find by symbol.

## 1. What the curation datastore does today (why metadata stays in OpenSearch)

OpenSearch holds **all** curation state, not only vectors. Per project an index set
`op_prj_<slug>__<role>` (`docs/opensearch_schema_design.md`), 1 shard, 0 replicas.

| Need | Where it lives in code | Vector-store fit |
|---|---|---|
| Per-document OCC with `if_seq_no`/`if_primary_term`, read-merge-write, human/locked preservation | `src/clients/occ.py` (`occ_update_one` `:113`, `occ_skip_on_conflict_bulk` `:186`, `occ_upsert_bulk` `:394`, `_merge_preserving_human` `:609`), `src/clients/occ_bulk.py`, `src/clients/occ_locks.py` | None of the candidates offers per-document compare-and-set on an arbitrary revision except pgvector (row-level transactions) |
| Guarded scripted updates (no-op on human-owned rows) | `src/services/curation/clustering/cluster_write_guard.py:100` | Not available as server-side scripts in Milvus/Qdrant/LanceDB |
| Aggregations: terms, filters, cardinality, composite (stats, review tabs, holdout strata, auto-promote) | `src/services/curation/stats_dataset_query.py`, `src/routers/curation/review.py`, `src/services/curation/holdout.py:199`, `src/services/curation/clustering/auto_promote.py:128` | Partial at best (Milvus/Qdrant: counts and facets only); pgvector: SQL |
| Sort, `search_after`, PIT, locate-in-queue by counting docs before a sort key | `src/routers/curation/review.py:428-525`, `src/services/curation/review_sorts.py`, `src/services/curation/clustering/embedding_reduce.py:280` | Limited sorting on payload; no PIT |
| Nested region boxes plus sibling nested box vectors | `region_boxes`, `region_box_embeddings` (`src/clients/curation_opensearch/bodies_core.py:167-218`) | Must be flattened to one entity per box |
| Dedup lookups by `imohash` via `msearch` | `src/services/curation/ingest.py:255-282` | Not a vector concern |
| Project isolation enforced at the transport (fail closed) | `src/services/projects/guard.py` (`check_request` `:341`, `ProjectGuardedTransport` `:488`), the client factory in `src/core/dependencies.py` | A second store needs its own guard |
| Additive mapping migrations on cold start | `src/clients/curation_opensearch/ensure_fields.py` | Per-engine schema evolution |
| Text search (OCR trigram, core routes) | `src/clients/opensearch/indexes.py` | Not a vector concern |

Conclusion fixed before measuring: **metadata stays in OpenSearch.** The only architecture
evaluated besides "OpenSearch only" is a **split**: vectors (plus a replicated set of filter fields)
in a vector store, everything else in OpenSearch. pgvector is the one candidate that could in
principle hold everything, but moving all of the above to SQL is a rewrite, out of scope for
v0.6.0; it is measured as an embedding index like the others.

Current vector configuration (all defaults, nothing tuned): every `knn_vector` is faiss HNSW,
`cosinesimil`, `m` 16 and `ef_construction` 512 (`OP_HNSW_M`, `OP_HNSW_EF_CONSTRUCTION`,
`src/clients/curation_opensearch/base.py:27-56`), float32, derived source on, no
`ef_search`, no `data_type`, no `mode`/`compression_level`, no `refresh_interval` setting.
Fields: items `pe_embedding` and `backbone_embedding` (1024), images `pe_embedding` (1024) and
`embedding` (512), nested `region_box_embeddings.embedding` (1024). Curation semantic search uses
efficient filtering inside the `knn` clause with `k = min(2000, page x page_size)`
(`src/services/curation/semantic_search.py:145-163,246`); the core `/search` route post-filters
(`src/clients/opensearch/search.py:187-215`), so a filtered core search can return fewer than k.
Near-duplicate decisions (cosine 0.98) do **not** use the k-NN index (exact numpy blocks,
`src/services/detection/frame_dedup.py`), and clustering reads all vectors by PIT slices, not by
k-NN. Measured storage: 8.85 KB per vector settled (v0.5.0 baseline, PR #206).

## 2. Workload model (what the benchmark must represent)

| Operation | Rate / shape today | Source |
|---|---|---|
| Insert | 7.4 vectors per image (6.4 crops + 1 whole frame) at policy `all`; 9 img/s now, 15-30 img/s expected after Wave 1 (**estimate**) = 70-220 vectors/s sustained, bulk per image | v0.5.0 baseline |
| Filtered kNN for review/search | `k` up to 2000, filters on validated, dismissed, excluded, test_holdout, cluster_id, class, rank, created_at, blur, near-dup | `semantic_search.py` `_build_filter` |
| Full-pool read for clustering | every residual vector, PIT + 8 slices, page 2000 | `embedding_reduce.py:280` |
| Metadata updates touching filter fields | full-pool `cluster_id` rewrite per retrain (12.8k docs per 2,000 images); auto-promote `class_validated`; human validation; exclude | `orchestrator.py:215-241`, `auto_promote.py` |
| Delete | project delete, item delete, combine copies | `src/services/projects/` |

The update row is the one that decides a split: every filter field the vector store needs must be
replicated, and clustering rewrites `cluster_id` for the whole residual pool on every retrain.

## 3. Datasets (public, pinned, real embeddings)

- **D100k.** PE crop and whole-frame vectors from ingesting COCO with policy `all`: 13,600 pinned
  images (`fetch_coco_subset.py --bench-set 13600`, same seed) give about 100k vectors at the
  measured 7.39 per image. Built after WP-1.1 so the vectors come from the shipped FP16 engine.
- **D1M.** All of COCO train2017 plus val2017 (about 123k images, about 0.9M vectors at 7.39 per
  image; the exact count is recorded). The pinned manifests keep only allowed licences; whether
  D1M may use every COCO image locally (only aggregate numbers published) is the owner's call
  (open question V1). If not, use the largest licence-filtered subset and state its size. Do
  **not** pad with jittered copies of D100k: synthetic structure makes recall meaningless.
- **Export format** (local, gitignored, `artifacts_local/bench/vectordb/<set>/`): `vectors.f32.npy`
  (L2-normalized, float32), `meta.parquet` with `item_id`, `image_id`, `kind` (crop | whole),
  `class_name`, `cluster_id` (IVF assignment from the live run), and synthetic seeded fields for
  filter selectivity: `project` (10 projects by hash, so a per-project filter keeps about 10 %),
  `class_validated` (10 %), `test_holdout` (5 %, the holdout rule's share), `review_dismissed` (2 %).
- **Queries.** 1,000 query vectors from images **not** inserted (held-out COCO val ids), plus
  1,000 in-set item vectors (the "more like this" case). Ground truth: exact cosine top-100 per
  query **per filter** with FAISS flat on GPU 0 (`artifacts_local/.../gt_<filter>.npy`).
- **Filters** (each with measured selectivity): none; project; project + class; project +
  `class_validated=false` + `test_holdout=false` + `review_dismissed=false` (the review default);
  project + `cluster_id` (about 0.2 % at K=512); a selectivity sweep at 50, 10, 1, 0.1 %.

## 4. Candidates and configurations

| Arm | Configurations |
|---|---|
| OS-shipped | exactly as `base.py` builds it today (faiss HNSW m16 efc512, float32, default ef_search) |
| OS-tuned | `ef_search` 64/128/256/512 (index setting or query `method_parameters`); m 16/32; faiss vs lucene engine; `data_type` byte / faiss SQ fp16 encoder; binary quantization and `mode: on_disk` with `compression_level` 8x/16x/32x plus rescoring; `refresh_interval` -1 during bulk then restore; bulk size 500/2000/5000; 1 vs 2 shards; force-merge to 1 segment before query runs; GPU-accelerated remote index build only if the shipped OpenSearch version supports it (verify, do not assume) |
| Milvus (standalone) | HNSW (m16, efc 512, ef 64-512), IVF_FLAT, IVF_SQ8, GPU_CAGRA (cuVS, GPU 0); scalar fields indexed for filters; partition key on `project` |
| Qdrant | HNSW (m16, ef_construct 512, ef 64-512), payload indexes on filter fields, int8 scalar quantization with rescoring, on-disk vectors |
| pgvector | `vector(1024)` and `halfvec(1024)`, HNSW (m16, ef_construction 512, `hnsw.ef_search` 64-512), iterative index scans for filtered queries (verify version support), B-tree indexes on filter columns |
| LanceDB (embedded) | IVF_PQ and IVF_HNSW_SQ, scalar indexes on filter columns, GPU index build if available |
| Reference | exact FAISS flat (CPU and GPU) and cuVS CAGRA standalone as upper bounds; not datastores |

Versions are pinned by image digest in the bench compose file and recorded with every result.

## 5. Harness

- Compose file `docker/bench/vectordb/compose.yml` (new), always run with
  `docker compose -p op060-vdb -f docker/bench/vectordb/compose.yml`, own ports, loopback only,
  GPU 0 only for GPU index builds and ground truth, one engine running at a time (memory isolation),
  same CPU and memory limits per engine (cgroup: 16 cores, 64 GB; heap/caches configured within that
  and recorded). Never the live `openprocessor` project.
- `scripts/bench/vectordb/` (new): `adapters/<engine>.py` implementing one interface
  `create(schema)`, `load(ids, vectors, meta, batch)`, `finish_load()` (refresh, flush, merge or index
  build), `search(q, k, filter) -> ids`, `update_fields(ids, field, values)`, `delete(ids)`,
  `stats() -> {bytes_on_disk, rss}`; `run.py` drives phases and writes JSON; `report.py` writes the
  tables. Pure logic (recall, percentile, selectivity) in `vdb_lib.py` with
  `tests/test_vdb_bench_lib.py`. Engine clients are imported only inside adapters and installed
  only in a bench venv or container, never added to `requirements.txt`.
- Repetitions: 3 per configuration, report median (min-max); queries warmed with 200 discarded
  queries.

## 6. Metrics

| Metric | Definition |
|---|---|
| Write throughput | vectors/s for the full load at 1 and 8 client threads; time until searchable (load + finish_load) |
| Streaming insert | vectors/s at a fixed 200 vectors/s offered load mixed with queries: query p95 impact |
| Query latency | p50, p95, p99 at concurrency 1, 8, 32 per filter, k = 10, 100, 2000 |
| Recall | recall@10 and recall@100 against the exact filtered ground truth; also "returned fewer than k" rate |
| Throughput | QPS at recall@10 >= 0.95 (pick the cheapest ef/nprobe that reaches it) |
| Storage | bytes on disk per vector after settle (merge/compaction, the storage doc's settle rule), RSS per vector at steady query load |
| Metadata update | time to rewrite `cluster_id` for 12.8k and for 100k vectors, and the query recall/latency during it |
| Delete | time to delete one project's 10 % and the space reclaimed |
| Operational weight | containers, idle RSS, idle CPU, backup/restore path, licence, GPU need |

## 7. Split architecture: the consistency costs, stated before measuring

If a vector store wins, the design would be: OpenSearch stays the source of truth for every item;
the vector store holds `item_id`, the vector and a **replica of the filter fields**.

1. **Dual write without a transaction.** Ingest writes the item doc (OpenSearch, OCC) and the
   vector (store). Needs an outbox: write the item with `vector_state: pending`, then upsert the
   vector, then flip the state; a reconciler repairs both directions after a crash. New failure
   modes: item without vector, vector without item (orphan), wrong vector version after a re-embed.
2. **Filter-field replication lag.** Every write that changes a replicated field (cluster rewrite of
   the whole pool, auto-promote, human validation, exclude, dismiss, holdout assignment) must also
   update the store. Until it lands, a filtered kNN can return an item a human just validated, or
   miss a newly residual item. Mitigation: over-fetch (k' = 2-4 k) and re-check candidates against
   OpenSearch by `mget` before returning; cost: one extra round trip and lower effective recall under
   selective filters. The benchmark measures both (section 6 "metadata update", and a re-check arm).
3. **The lock rule is unaffected** (the store never decides a class or box), but review queues and
   counts computed in OpenSearch and lists computed from the store can disagree during lag.
4. **Isolation.** One collection per project, named from the bound project; a guard equivalent to
   `ProjectGuardedTransport` for the store client (fail closed on any cross-project name), and the
   factory rule (no raw client construction) extended to it. Project delete, slug retirement and
   combine must cover the store.
5. **Operations.** A second stateful service to back up, upgrade, monitor and size; restores must be
   consistent with the OpenSearch snapshot (or rebuilt from it: vectors are re-derivable from
   OpenSearch only if OpenSearch still stores them, which removes the storage win).
6. **Region boxes** become one entity per box with `parent_item_id`.

## 8. Decision criteria

Adopt a split for the embedding index only if **all** hold at D1M on the benchmark host:

1. On the review-default filter and the cluster filter, p95 latency at recall@10 >= 0.95 is at least
   **2x better** than the best OS-tuned configuration, or storage plus RAM per vector is at least
   **40 % lower** at the same recall than the best OS quantized configuration.
2. The `cluster_id` rewrite of 100k vectors completes in under 60 s without dropping recall@10 below
   0.9 during the rewrite.
3. The consistency design of section 7 is implementable with a reconciler test and an isolation
   guard test before any production use (separate plan, owner yes).
4. Operational weight is acceptable to the owner (idle RSS and a backup story recorded).

Otherwise: adopt the best OS-tuned configuration (expected outcome is unknown; the arms are run
precisely to find out), record its numbers in `docs/PERFORMANCE.md` and
`docs/design/storage_sizing_and_ingest_baselines.md`, and close #56 with them. Any change to the
shipped index settings (ef_search, quantization) is its own WP with the parity gates of the wave
plan (kNN overlap@10 against the float32 reference >= 0.95 on the review-default filter).

## 9. Work breakdown (issue #221)

| Step | Output | Tier |
|---|---|---|
| V.1 dataset export and exact ground truth (D100k first) | `artifacts_local/bench/vectordb/D100k/` | Sonnet |
| V.2 harness, OS-shipped and OS-tuned adapters, tests | `scripts/bench/vectordb/`, `tests/test_vdb_bench_lib.py` | Sonnet |
| V.3 OS arms on D100k; publish the #56 table | `docs/PERFORMANCE.md` section | Sonnet |
| V.4 other engines' adapters, D100k runs | JSON + tables | Sonnet |
| V.5 D1M build (after Wave 1) and the full matrix for the top three arms | JSON + tables | Sonnet |
| V.6 decision memo against section 8, appended to this file | this file, section 10 | Opus |

## 10. Results

Empty until V.3. Record the hardware, versions and digests with every table.

## 11. Open questions for the owner

| ID | Question | Recommendation |
|---|---|---|
| V1 | May D1M use all COCO train2017 images locally (only aggregate numbers published), or only licence-filtered images? | All locally, aggregates only; otherwise state the smaller size |
| V2 | Is pgvector as a full replacement for OpenSearch (metadata too) in scope for a later milestone? | Not in v0.6.0; reconsider only if the split wins and the owner wants one datastore |
| V3 | Should the OS-tuned winner (for example a set `ef_search`) ship in v0.6.0 if it passes the gates? | Yes, as its own small WP |
