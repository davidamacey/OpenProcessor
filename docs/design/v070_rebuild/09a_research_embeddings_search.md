# Embeddings and semantic search research plan: encoders, dual-field swap, search, kNN at scale

| Field | Value |
|---|---|
| Superseded parts | the landing lanes, waves and file ownership of this plan's section 8, its GPU lock and its GPU-hour estimates are superseded by the program (review RR-16, owner answers of 2026-11-04): lanes in [03d_stage4_job_families.md](03d_stage4_job_families.md) (W2.3a and W2.3b), [03b_stage2_control_data.md](03b_stage2_control_data.md), [03f_stage6_scheduler_scaleout.md](03f_stage6_scheduler_scaleout.md), [05_b4_stateless_api.md](05_b4_stateless_api.md) and [06e_w5e_plate_text.md](06e_w5e_plate_text.md); one lock `gpu<N>.lock` per card shared by every topic ([traceability.md](reference/traceability.md) X-14); short tiered runs under a cap of about 10 GPU hours before the tag ([v070_rebuild/08_research_prototype_track.md.md](08_research_prototype_track.md) section 2.1). A fresh agent loads those files first |
| Status | plan, research. Nothing here is implemented. Written 2026-11-04 against `origin/main` at `49731595`. Section 5 (the seams) is the only part meant for the v0.7.0 rebuild; every experiment runs after the v0.7.0 tag (MQ-9) except the docs-only registration EP-0 |
| Depends on | the v0.7.0 foundation: PostgreSQL for control data, Celery and Valkey for every long job, a stateless API ([v070_rebuild/02_control_plane_database.md.md](02_control_plane_database.md), [02a](02a_postgres_schema.md), [02b](02b_jobs_locking_events.md)); OpenSearch keeps vectors and search (D-40); every service needs PostgreSQL, Valkey, OpenSearch and Triton (D-69); B5's decode-once path for any new image encoder |
| Issues | #347 (second encoder evaluation), #365 (encoder swap with a dual-field migration), #435 (frame target), #436 (`group_by=image`); results of closed #414 (search latency at 100k and 1M) and #402 (similar, image and box search) |
| Absorbs | `docs/design/issue_plans/issue-347-plan.md` (deleted in this commit; its stages, gates, bars, traps and judgment calls are carried in sections 6 and 10). Keeps and references [embedding_model_eval_plan.md](../../research/embedding_model_eval_plan.md) (the model facts and sources for the #347 candidate, scenarios S1 to S7, the TensorRT ladder, the metrics schema) |
| Owner rules applied | #347 (2026-10-29): a new encoder ships only as an optional encoder, off by default, never replacing PE-Core unless it proves better; text encoder CPU or GPU selectable. 2026-11-04: clean backend rewrite, research to best practice first, prototypes are simple local-venv GPU scripts on set P plus public image and vector datasets, plan first and run nothing |
| Audience | a fresh agent with no conversation memory. Claims about the tree cite `path:line` at the commit above; re-find by symbol. Claims about models are vendor claims unless measured |

In this file the private high-resolution photo set is "set P". The second public encoder under
evaluation in #347 is named by its public name, EmbeddingGemma 2, only where the text must identify
it; elsewhere it is "the #347 candidate". Arm keys and code never spell it (section 6.1).

---

## 1. Plain summary

- **What exists.** One image-text encoder per vector field, wired by constant: PE-Core-L14-336
  (1024-d) in the curation `pe_embedding` fields (items, images, per-box nested vectors), and
  MobileCLIP2-S2 (512-d) in the core inference indexes. Nothing records which model wrote a vector:
  combine compares only the dimension (`src/services/projects/combine/warnings.py:46-50`, "Items
  carry no encoder id, so the dimension is the whole check"). 40 files under `src/` and `scripts/`
  name the field constant or the literal (97 references, `rg -c "pe_embedding|ITEM_EMBEDDING_FIELD"`).
- **What is measured.** Search is fast enough at the sizes we tested: curation text search p99
  86-177 ms at 100k items and 87-130 ms at 964k (jittered) items, under a 300 ms bar; the text encode
  (about 53 ms, CPU onnxruntime in Triton) is the largest stage, not the kNN (#414). OpenSearch tuned
  ties every dedicated vector store at 100k real vectors; the Lucene engine is already the default
  (#221). PE-Core-L stays the encoder: no challenger passed the WB2 bar (PE-Core-G is better by
  0.01-0.04 Recall@1 at 8x the cost). The PE whole-image text search row, the #347 candidate, any text
  embedding of OCR or captions, and any 1M-vector run on real vectors are **not measured**.
- **What the rebuild must include (section 5).** An embedding registry in PostgreSQL (model
  versions, per-project embedding spaces, re-index runs, evaluation runs); every reader resolves its
  vector field through the bound project's space instead of a constant; one Celery job kind,
  `embedding_reindex`, that fills a new field next to the old one with progress, pause, cancel and
  resume; a fenced primary flip; query paths that name their space and refuse a mismatched one.
  The 0.6.0 field names stay as data in the registry (no reindex in the upgrade). This is the
  machinery #365 needs; with it in v0.7.0, v0.8.0 F9 shrinks to "register a candidate and flip".
- **What the research decides (section 6).** Whether the #347 candidate or another public encoder
  earns an optional space (OCR text, captions, a second image view, a clustering view); whether the
  PE frame vector is good enough for whole-image search (decides #435); index settings for 1M real
  vectors (Lucene engine, quantization, OpenSearch 3.9 `half_float`); whether the text encode moves to
  the GPU or behind a shared Valkey cache.
- **What it does not do.** No second vector database (measured: no gain worth a split), no vectors
  in PostgreSQL (D-40), no in-place model overwrite of the Immich kind, no change of the default
  encoder without the registered bars and the owner's yes.

## 2. Measured today (every number with its source)

### 2.1 Search quality (public COCO val2017, 2,000 images, seed 20261028)

Source: [retrieval_baseline.md](../../research/retrieval_baseline.md); raw JSON in
`docs/benchmarks/retrieval_baseline/`; sample pinned in
`docs/benchmarks/retrieval_baseline/sample_ids.json`.

| Path | Queries | recall@1 | recall@5 | recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| core text-to-image, MobileCLIP, whole image | 2,000 (first caption) | 0.567 [0.546, 0.588] | 0.817 | 0.886 [0.872, 0.901] | 0.671 | 0.723 |
| core, MobileCLIP | 10,000 (5 captions) | 0.591 [0.581, 0.600] | 0.824 | 0.896 [0.890, 0.902] | 0.691 | 0.740 |
| curation text-to-crop, PE-Core, ranked by first crop of the image | 2,000 | 0.383 [0.361, 0.403] | 0.628 | 0.733 [0.715, 0.752] | 0.488 | 0.547 |
| curation, PE-Core crops | 10,000 | 0.401 [0.392, 0.411] | 0.642 | 0.738 [0.729, 0.746] | 0.504 | 0.560 |
| core image-to-image, category Jaccard >= 0.5 proxy | 1,982 | precision@10 0.476 (normalised 0.507, chance 0.015) | | | | |
| **PE whole-image text-to-image on the same sample** | | **not measured** ("the experiment that separates them ... (not done)") | | | | |

The two text rows are a product-path comparison (crops against whole images, two encoders), not an
encoder comparison. Vendor anchor for PE: the PE-Core-L14-336 card reports COCO text-to-image 57.1
zero-shot ([HF card](https://huggingface.co/facebook/PE-Core-L14-336); G/14-448 58.1, B/16 50.9),
which is a whole-image number on the standard 5k split, so EP-2 must reproduce it before any PE row
is read.

Near-duplicate grouping at ingest (same note): at the default cosine 0.99, JPEG q60 variants group
34/100 and half-size 13/100; at 0.95 nearly all transforms group but distinct images mix in; 2,000
distinct originals gave one false group at 0.99 (similarity 0.9924).

### 2.2 Encoder quality and cost on labelled benchmarks (WB2, whole images, local-only sets)

Source: [wb2_standard_benchmarks.md](../../research/wb2_standard_benchmarks.md) sections 9.2-9.6
(registered bars; sanity check not satisfied as registered, so comparisons are exploratory).

| Model | CARS196 R@1 | CUB R@1 | SOP R@1 | img/s (batch 32, fp16, shared GPU) | peak MiB | dim | re-embed 1M (h) |
|---|---|---|---|---|---|---|---|
| PE-Core-L (letterbox) | 0.961 | 0.867 | 0.796 | 133 | 1,835 | 1024 | 2.1 |
| PE-Core-G (letterbox) | 0.969 | 0.907 | 0.820 | 17 | 9,170 | 1280 | 15.9 |
| MobileCLIP2-S2 (centre) | 0.964 | 0.752 | 0.674 | 815 | 629 | 512 | 0.34 |
| SigLIP 2 So400m (letterbox) | 0.975 | 0.744 | 0.790 | 314 | 1,982 | 1152 | 0.88 |
| DINOv2-L (centre) | 0.825 | 0.892 | 0.576 | 373 | 1,451 | 1024 | 0.74 |

B1 (gain over PE-Core-L): every challenger fails; PE-Core-G +0.008 / +0.040 / +0.024 fails on CARS196
(at the ceiling). Letterbox is not a lever on whole images (B2 fails for all five), unlike on set C
crops. Text search was not measured in WB2. The earlier set C ranking (small encoders ahead of
PE-Core-L) did not reproduce on author-labelled sets
([fine_grained_benchmark_results.md](../../research/fine_grained_benchmark_results.md),
[labelled_benchmarks.md](../../research/labelled_benchmarks.md)).

### 2.3 Encoder serving

| Number | Value | Source |
|---|---|---|
| PE image engine, TensorRT FP16, one Triton instance, RTX 3080 Ti | 29.8 crops/s at batch 1, **90.2 at batch 4**, about 77 above; batched output bit-identical to batch 1 | [pe_encoder_batching_benchmark.md](../../research/pe_encoder_batching_benchmark.md) |
| PE image engine INT8 on Ampere (#213), FP16 engine pass (#227), multi-GPU PE (#228) | as recorded | [PERFORMANCE.md](../../PERFORMANCE.md) sections "INT8 evaluation", "PE image encoder engine pass", "Multi-GPU PE encoding" |
| PE text encode, Triton `pe_text_encoder` (onnxruntime, one CPU instance), uncached | about 53 ms p50, p99 up to 148 ms under host load; cached 0.3 ms | PERFORMANCE.md "Search latency at 100k and 1M items" |
| Same, earlier campaign on set P (91,184 items) | about 32 ms extra per uncached query; the per-worker LRU (256 x 32 workers) hit rate is far below what 256 suggests | [v070_measurement_log.md](../../research/v070_measurement_log.md) section D |
| In-process ONNX text session per uvicorn worker | container RSS about 84 GiB with 32 workers; removed from `auto` | `src/clients/pe_encoder.py:29-36` |

### 2.4 Search latency at scale (#414, closed; public COCO items)

Source: PERFORMANCE.md "Search latency at 100k and 1M items". 10,000 COCO images ingested (64,455 real
items); the 100,000 and 964,455 item indexes were grown by copying real documents with **jittered PE
vectors** (sigma 0.01, cosine about 0.95 to the original): real class mix and filters, synthetic
geometry. Lucene HNSW m 16, ef_construction 128, 16 segments. 600 measured distinct queries, 3 runs,
loaded host.

| Size | Condition | encode p50 | kNN p50 / p99 | served p50 / p99 |
|---|---|---|---|---|
| 100,000 | unfiltered | 53.8 | 8.1 / 15.4 | 69.9 / 169.5 |
| 100,000 | `person` 32.2 % | 54.7 | 12.0 / 26.0 | 76.3 / 176.8 |
| 964,455 | unfiltered | 53.1 | 7.4 / 12.5 | 69.6 / 86.9 |
| 964,455 | `bottle` 2.7 % | 53.3 | 29.3 / 35.6 | 93.2 / 108.8 |
| 964,455 | `car` 5.4 % | 53.9 | 47.2 / 58.5 | 110.7 / 129.9 |
| 964,455 | repeated query (cache hit) | 0.3 | 8.7 / 12.8 | 21.0 / 28.1 |

Findings carried: no request over 300 ms in 18,000; the filtered traversal cost grows with
restrictiveness at 1M, not with size; HNSW `m` 32 was not needed. Filtered kNN on a set-P copy (91,184
items, Lucene, #384): unfiltered 4/7 ms p50/p95, `car` (15 %) 22/30 ms; `m` 32 gave 19 ms (not adopted);
the filter is applied inside the HNSW traversal on both engines ([retrieval_baseline.md](../../research/retrieval_baseline.md)
"Filtered kNN latency"). Synthetic 1M crops (seeded mixture, centroids from set P): unfiltered p99 10.6
ms, 0.74 % filter p99 12.2 ms (v070_measurement_log.md "k-NN query latency"). **No 1M row on real,
independent vectors exists.**

### 2.5 Vector store and index settings (#221, #56; 100,000 real PE vectors of COCO, no 1M)

Source: PERFORMANCE.md "Vector store evaluation" (raw `docs/benchmarks/vectordb/`), protocol
[v060_vector_db_evaluation_plan.md](../../research/v060_vector_db_evaluation_plan.md).

| Arm (review-default filter, 8.3 %) | recall@10 | p95 ms | QPS c8 | disk KiB/vec | RAM KiB/vec |
|---|---|---|---|---|---|
| OpenSearch faiss as shipped then (13 segments) | 1.000 | 9.6 | 336 | 8.20 (engine 10.99) | 6.99 |
| OpenSearch faiss merged, ef_search 100 | 0.967 | 8.1 | 423 | 8.20 | 7.72 |
| **OpenSearch Lucene** | 0.997 | 8.8 | 403 | **4.10** | **2.87** |
| OpenSearch faiss `on_disk` 16x, ef_search 128 | 0.945 | 7.8 | 388 | 4.45 | 3.18 |
| Milvus 2.6.25 HNSW | 0.954 | 9.8 | **741** | 23.44 | 6.32 |
| Qdrant 1.19.2 HNSW | 1.000 | 7.3 | 293 | 10.71 | 0.76 |
| pgvector 0.8.7 `vector` / `halfvec` | 1.000 / 0.988 | 52.2 / 9.5 | 131 / 903 | 22.57 / 10.30 | 13.56 / 5.56 |
| LanceDB IVF_PQ | 0.977 | 33.9 | 228 | 8.50 (engine) | n/a |

Also measured: merged faiss falls to recall 0.912 at a 2.6 % filter (Lucene 1.0); pgvector p95 145 ms
at a 0.1 % filter (planner switches paths); `cluster_id` rewrite of 100k vectors in place 35-52 s
(Milvus 12 s, Qdrant 149 s, pgvector 202 s); OpenSearch query p95 doubles under 200 inserts/s (Milvus
flat). Native graph memory of faiss is 4.15 KiB/vector measured against the `4 x d + 8 x m` formula;
Lucene uses page cache, not native memory. Extrapolated (not measured) to 7.4M vectors of a 1M-image
project: faiss 61 GB disk and 31 GB native memory, Lucene 30 GB and 0. Decision then: stay on
OpenSearch; Lucene engine for new projects (since shipped: `src/clients/curation_opensearch/base.py:27-49`,
`ef_construction` 128 and `m` 16 from `src/config/curation.py:159-160`); do not force-merge; leave
`ef_search` alone.

### 2.6 Storage

[storage_sizing_and_ingest_baselines.md](../../research/storage_sizing_and_ingest_baselines.md): a 1024-d
float32 vector costs 8.4 KB on faiss (graph plus flat copy), about 4.5 KiB per item on Lucene; derived
source must stay on (25 KB per vector with it off); 0.5.0 baseline 8.83 KB per vector and 65.4 KB per
image at 6.4 items per image. Re-measured reprocess throughput on set P (#259) is in
v070_measurement_log.md sections E1 and F1.

### 2.7 What exists in the code (read at `49731595`)

| Fact | Where |
|---|---|
| Item vector field constant `ITEM_EMBEDDING_FIELD = 'pe_embedding'`; dims `embedding_dim` 512, `encoder_embedding_dim` 1024; `hnsw_ef_construction` 128, `hnsw_m` 16 | `src/config/curation.py:44`, `:156-160` |
| `_knn_field`: Lucene HNSW, cosine, per-call dim | `src/clients/curation_opensearch/base.py:27-49` |
| Images index: 512-d `embedding` and 1024-d whole-frame `pe_embedding`; items `pe_embedding`; nested `region_box_embeddings` {`box_id`, `bbox_norm`, `embedding`} as a sibling of `region_boxes` | `src/clients/curation_opensearch/bodies_core.py:97`, `:105`, `:212-225`, `:418` |
| `embedding_state` (embedded, not_selected, deferred, failed) keyed to the one field; the `exists` test is authoritative | `src/domain/embedding_state.py:1-55` |
| Core indexes `visual_search_{global,vehicles,people,faces,ocr}`, MobileCLIP 512-d (faces ArcFace 512-d) on **faiss** | `src/clients/opensearch/names.py:18-25`, `src/clients/opensearch/indexes.py:96-130` |
| Curation search: `GET .../search/text`, `GET .../search/similar`, `POST .../search/image`, `target=item|box`; filtered kNN inside the `knn` clause, `k = min(2000, page x page_size)`; kNN and the unembedded count run concurrently (B4 lane 4A-5 turns them into one `_msearch`) | `src/routers/curation/search.py`, `search_image.py`, `src/services/curation/semantic_search.py:52`, `:246`, `:279-283` |
| PE text backends `triton` (default), `torch` and `onnx` in-process fallbacks selected by `OP_PE_TEXT_BACKEND` | `src/clients/pe_encoder.py:14-42`, `src/clients/pe_text_backends.py` |
| Per-request stage timing (`Server-Timing`, `op_search_stage_seconds`) | `src/utils/search_timing.py` (#414) |
| Combine warns by dimension only, drops mismatched vectors and defers those items | `src/services/projects/combine/warnings.py:42-60` |
| Research harnesses to reuse: retrieval metrics with bootstrap (`scripts/bench/retrieval_eval_metrics.py`, tested), the latency subcommand (`scripts/bench/retrieval_eval_latency.py`), the vector-store harness (`scripts/bench/vectordb/`, `docker/bench/vectordb/compose.yml`), WB2 scoring (`scripts/research/standard_benchmarks/`), cluster scoring (`scripts/bench/cluster_eval.py`), run envelope (`scripts/bench/scorecard.py`), item replication (`scripts/bench/replicate_items.py`) | as named |

The images-index 512-d `embedding` field has no writer in `src/services/curation/ingest_index.py`
(`rg -n "embedding" src/services/curation/ingest_index.py`). Lane ES-A confirms with `rg` over the
whole tree; if nothing writes it, the rebuild drops it from `_images_body` (no dead field) and the
registry does not list it.

## 3. Literature and industry review

### 3.1 Encoders (model cards and papers; vendor numbers are vendor numbers)

| Model | Facts that matter here | Source |
|---|---|---|
| PE-Core L14-336 (production), G14-448, B16-224 | One CLIP-style vision-text pair, attention pooling; G is the main checkpoint, L and B distilled from it; COCO T2I zero-shot 57.1 (L), 58.1 (G), 50.9 (B); Apache-2.0 weights | [paper arXiv 2504.13181](https://arxiv.org/abs/2504.13181), [card](https://huggingface.co/facebook/PE-Core-L14-336), [repo](https://github.com/facebookresearch/perception_models) |
| SigLIP 2 (B, L, So400m, g; NaFlex variants) | Sigmoid loss plus captioning and self-distillation; multilingual; better retrieval than SigLIP at every size; Immich lists So400m-16-384 and B-16 SigLIP 2 as Pareto-optimal for English smart search (recall about 86 % and 85 %, 57 ms and 5.8 ms, estimates) | [arXiv 2502.14786](https://arxiv.org/abs/2502.14786), [Immich searching docs](https://docs.immich.app/features/searching/) |
| MobileCLIP2 (S0-S4, B, L) | Low-latency family; S4 matches SigLIP-So400m/14 ImageNet zero-shot at half the size; the core path ships S2 | [arXiv 2508.20691](https://arxiv.org/abs/2508.20691), [collection](https://huggingface.co/collections/apple/mobileclip2) |
| DINOv3 (and dino.txt) | Self-supervised, strongest on dense and fine-grained image similarity; text alignment only through a separately trained dino.txt head; custom DINOv3 licence (read before any use) | [arXiv 2508.10104](https://arxiv.org/abs/2508.10104), [repo](https://github.com/facebookresearch/dinov3) |
| EmbeddingGemma 2 (the #347 candidate) | 740M total, text 270M plus optional 170M vision tower, one 768-d space, Matryoshka 768/512/256/128, task prefixes required, **bf16 or fp32 only (fp16 overflows)**; vendor MMEB v2 Image Hit@1 57.28, MIEB lite 65.67; no COCO number, no named rivals, no server latency | [embedding_model_eval_plan.md](../../research/embedding_model_eval_plan.md) section 1 (all sources there) |
| Qwen3-VL-Embedding 2B / 8B | LLM-based, 2048 / 4096-d, Matryoshka; MMEB-V2 Image 74.96 (2B) and 80.12 (8B) on the leaderboard; Apache-2.0; vLLM or transformers, no TensorRT path | [arXiv 2601.04720](https://arxiv.org/abs/2601.04720), [MMEB leaderboard](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard) |
| Nemotron 3.5 Embed 1B / 8B | LLM-based multimodal, 2048-d (truncatable), OpenMDW-1.1 licence; the omni 3B is research-only | eval plan section 5.2 |
| jina-clip-v2 | 865M, 89 languages, Matryoshka 1024 to 64; **CC BY-NC 4.0** on the card, so not shippable here | [card](https://huggingface.co/jinaai/jina-clip-v2) |
| Text-only embedders for OCR lines and captions | EmbeddingGemma (v1, 308M, text), Qwen3-Embedding 0.6B, BGE-M3: small, multilingual, ranked on MTEB | [EmbeddingGemma paper arXiv 2509.20354](https://arxiv.org/abs/2509.20354), [Qwen3-Embedding arXiv 2506.05176](https://arxiv.org/abs/2506.05176), [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) |

Leaderboards. MTEB (text) and MIEB (image and image-text, 130 tasks, ICCV 2025,
[arXiv 2504.10471](https://arxiv.org/abs/2504.10471)) run from the maintained `mteb` package; results
live in [embeddings-benchmark/results](https://github.com/embeddings-benchmark/results). MIEB's authors
find "no single method dominates across all task categories". MMEB-V2 (TIGER-Lab) is dominated by
LLM-based embedders of 2B-8B parameters. Neither leaderboard has a detector-crop task, so a leaderboard
rank says little about curation crops; that is why WB2's author-labelled sets and our own crop sets
decide, and why the leaderboards are used only as a loader-sanity anchor (reproduce one listed task
within 1 point, gate A4).

### 3.2 How comparable systems handle a model change

| System | Design | Lesson |
|---|---|---|
| Immich smart search | One active CLIP model; Postgres with VectorChord; changing the model resizes the `smart_search` vector column to the new dimension and requires re-running the Smart Search job on all assets; the docs warn that old incompatible data can remain and the fix is to revert and repeat ([searching](https://docs.immich.app/features/searching/), [issue 9398](https://github.com/immich-app/immich/issues/9398)) | In-place, single-field swap: search is broken or mixed while the job runs, and a failure leaves mixed spaces. We want the opposite: write the new space beside the old, flip once, keep the old until the flip is confirmed |
| OpenSearch k-NN | A `knn_vector` field's engine, method and dimension are fixed at creation; new fields can be added with put-mapping; mapped fields cannot be removed ([knn-vector](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-vector/), [put mapping](https://docs.opensearch.org/latest/api-reference/index-apis/put-mapping/)) | A space is a new field. Retiring a space clears values (disk reclaimed on merge), the mapping line remains; the registry is what says the field is dead |
| Index-per-version with an alias swap | The usual search-engine pattern for a full mapping change ([reindex](https://docs.opensearch.org/latest/api-reference/document-apis/reindex/), [aliases](https://docs.opensearch.org/latest/im-plugin/index-alias/)) | Our items index also holds OCC-guarded review state that keeps changing during a long re-embed; copying the whole document set and freezing writes, or replaying them, is a bigger risk than adding one field |
| Dedicated vector store split | Our own #221 section 7: dual write, filter-field replication lag, a second isolation guard | Rejected on measured numbers (section 2.5) |

### 3.3 OpenSearch k-NN facts this plan relies on (docs read 2026-11-04)

- Engines: Lucene HNSW (native filtering), faiss (HNSW, IVF, more compression), nmslib deprecated,
  JVector by plugin. Lucene defaults `m` 16, `ef_construction` 100 and **ignores `ef_search`** (it
  uses `k`); faiss defaults `ef_search` 100
  ([methods and engines](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-methods-engines/)).
- `k` is per segment for faiss and per shard for Lucene, maximum 10,000; `size` is the final count
  ([approximate k-NN](https://docs.opensearch.org/latest/vector-search/vector-search-techniques/approximate-knn/)).
- Filtering: Lucene chooses exact pre-filtered or approximate search from N, P and k; faiss uses
  `knn.advanced.filtered_exact_search_threshold`, an exact fallback when the filtered approximate
  search returns fewer than k (3.5+), and ACORN-style exploration with memory-optimized search
  below 60 % filter share (3.1+)
  ([efficient filtering](https://docs.opensearch.org/latest/vector-search/filter-search-knn/efficient-knn-filtering/)).
  Lucene itself has an ACORN-1-style filtered HNSW search since 10.2 (default threshold 60 %,
  [lucene#14160](https://github.com/apache/lucene/pull/14160)); OpenSearch 3.6 is on Lucene 10.4.
- Compression and modes: `mode` `in_memory` (default) or `on_disk`; `compression_level` 1x to 32x,
  with Lucene supporting 1x, 4x, 8x, 16x and 32x and faiss 2x to 32x; `on_disk` rescoring
  `oversample_factor` 2.0 at 32x; `on_disk` with 1x turns on memory-optimized search
  ([memory-optimized vectors](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-memory-optimized/),
  [disk-based search](https://docs.opensearch.org/latest/vector-search/optimizing-storage/disk-based-vector-search/)).
  **`half_float` vectors are new in 3.9**, and 3.9 adds 2-bit and 4-bit scalar quantization on both
  engines ([3.9 release notes](https://github.com/opensearch-project/opensearch-build/blob/main/release-notes/opensearch-release-notes-3.9.0.md));
  1-bit scalar quantization for Lucene arrived in 3.6
  ([3.6 release notes](https://github.com/opensearch-project/opensearch-build/blob/main/release-notes/opensearch-release-notes-3.6.0.md)).
  We run 3.6.0 (`docker-compose.yml`, `OPENSEARCH_IMAGE` default).
- Search tuning: few segments per shard, warm up native indexes, return ids and scores only or
  exclude vectors from `_source`, `docvalue_fields` for bulk vector reads (3.7+)
  ([performance tuning](https://docs.opensearch.org/latest/vector-search/performance-tuning-search/)).
- Hybrid lexical plus vector: the `hybrid` query with a search pipeline; reciprocal rank fusion
  through `score-ranker-processor` (`rank_constant` default 60, since 2.19); the docs report RRF
  3.86 % lower nDCG@10 than score normalization on BEIR at similar latency
  ([RRF](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/rrf/),
  [score ranker](https://docs.opensearch.org/latest/search-plugins/search-pipelines/score-ranker-processor/)).
- Collapse for one row per image: keyword field with doc values, `search_after` paging, not with
  rescore ([collapse](https://docs.opensearch.org/latest/search-plugins/searching-data/collapse-search/)).

### 3.4 Vector databases (comparison only)

pgvector (HNSW and IVFFlat, `halfvec`, iterative index scans for filtered queries since 0.8,
[README](https://github.com/pgvector/pgvector)); Qdrant (filterable HNSW with payload indexes,
quantization, [indexing docs](https://qdrant.tech/documentation/concepts/indexing/)); Milvus (HNSW, IVF,
GPU CAGRA, partition keys, [index docs](https://milvus.io/docs/index-explained.md)); VectorChord (the
Postgres extension Immich uses). Our measurement (section 2.5) already ranks them on our vectors and
filters at 100k; the only clear Milvus lead is concurrent QPS (1.7x) and flat latency under inserts.
Nothing in the literature changes the 0.6.0 decision; the open item is a real 1M point.

### 3.5 Public vector and retrieval benchmarks

| Set | What it gives | Use here | Source |
|---|---|---|---|
| ann-benchmarks | recall against QPS at k 10 on classic sets (glove, sift, gist-960, fashion-mnist); no filters, no CLIP-type sets | index-behaviour sanity only; not representative of our vectors | [ann-benchmarks.com](https://ann-benchmarks.com/) |
| VIBE | modern embedding sets: `imagenet-clip-512-normalized` (1,281,167 x 512, in-distribution), **`laion-clip-512-normalized` (1,000,448 x 512, text queries against image vectors, out-of-distribution)**, `landmark-dino-768-cosine` (760,757) | the public 1M text-to-image index rung (OOD queries are exactly our text search) | [arXiv 2505.17810](https://arxiv.org/abs/2505.17810), [repo](https://github.com/vector-index-bench/vibe) |
| big-ann NeurIPS'23 filtered track | YFCC 10M CLIP 192-d uint8 with tag filters, 100,000 queries | filtered-search behaviour at 1M (a slice) with real tag selectivity | [big-ann-benchmarks](https://big-ann-benchmarks.com/neurips23.html) |
| VectorDBBench | Cohere 1M / 10M 768-d with 1 % and 99 % filter cases, LAION 100M | external cross-check of engine rankings only | [repo](https://github.com/zilliztech/VectorDBBench) |
| COCO 5k (Karpathy test), Flickr30k | standard image-text retrieval protocol; COCO is what the PE card reports | quality anchor (gate A6); Flickr30k images are research-only, local rows only | COCO terms; card |
| BEIR / MTEB retrieval tasks | text retrieval nDCG@10 | loader sanity for text arms (gate A4); OCR strings are not BEIR-like, so BEIR never decides placement A | [mteb](https://github.com/embeddings-benchmark/mteb) |

## 4. Candidate designs

Legend: ✅ adopt, ❌ reject (with the reason), 🟡 decide by a prototype in section 6.

### 4.1 Architecture

| ID | Design | Verdict | Why |
|---|---|---|---|
| A1 | **Embedding registry in PostgreSQL; a space per (project, target, model version) with its own vector field in the existing items or images index; readers resolve the field through the bound project** | ✅ | identity of a vector is the space, never the dimension; matches "class identity by name"; no second store |
| A2 | Keep the 0.6.0 physical names (`pe_embedding`, `embedding_state`, `region_box_embeddings`) as registry data for the spaces the upgrade registers; new spaces get `vec_<space_key>` names | ✅ (J2) | renaming in the one-step upgrade means re-writing every vector of every project; the registry already removes the constant, so the old names are data, not a shim |
| A3 | Dual-field swap: build the new space beside the old with a Celery job, flip the primary in one fenced transaction, clear the old space after a grace period | ✅ | search never breaks or mixes spaces; resumable; matches #365 |
| A4 | Index per model version with an alias swap | ❌ | copies the OCC-guarded review state of a live project; replaying writes made during a multi-hour re-embed is a correctness risk |
| A5 | In-place overwrite of one field (Immich) | ❌ | mixed spaces during the job; dimension change impossible on a fixed mapping |
| A6 | Separate vector database (Milvus, Qdrant, LanceDB) beside OpenSearch | ❌ now, revisit only on the section 6.6 trigger | measured tie at 100k (section 2.5); certain consistency cost |
| A7 | Vectors in PostgreSQL (pgvector or VectorChord) now that PostgreSQL is in the stack | ❌ | D-40; pgvector p95 52 ms on the review filter and 145 ms at 0.1 %; the metadata, OCC and aggregations stay in OpenSearch |
| A8 | Encoder identity checked at query time: a query vector carries its space key, the query builder refuses a field of another space | ✅ | the "never mix spaces" rule as code, fail closed |
| A9 | Per-space HNSW settings fixed in the registry (engine, m, ef_construction, mode, compression, data type); a settings change is a new space | ✅ | a knn field's method is immutable in OpenSearch, so the registry must say what it is |
| A10 | Engine rebuilds (B3 pin bumps, a new TensorRT) keep the model version only if the engine-parity gate passes (min cosine >= 0.999 against the reference on the pinned images); otherwise a new version and a re-index | ✅ | silent drift of an encoder behind an unchanged name is the failure the registry exists to stop |

### 4.2 Encoders

| ID | Candidate | Verdict | Why |
|---|---|---|---|
| E1 | PE-Core-L14-336 stays the primary image-text space | ✅ | WB2 B1: no challenger passes; text tower; TensorRT FP16 path proven |
| E2 | MobileCLIP2-S2 stays the core inference API's space | ✅ | fastest; core indexes are deployment-wide; the core path's text search is the better product row today (section 2.1) |
| E3 | PE-Core-G as a primary | ❌ | +0.01-0.04 R@1 at 8x encode and 5x memory, fails B1; 🟡 as an optional "quality" space for small projects only if the owner wants it (J9) |
| E4 | The #347 candidate as an optional OCR-text or caption space (placements A, B) | 🟡 EP-4 stage B | text is where the product has nothing semantic today (OCR is keyword-only) |
| E5 | The #347 candidate's vision tower as a second image view or clustering view (placements C, D) | 🟡 EP-4 stage C | complement-only by owner rule |
| E6 | SigLIP 2 So400m or MobileCLIP2-S4 as an optional text-search image space | 🟡 EP-4 (screen only) | strong public text-to-image results; CUB loss in WB2 says not a clustering space |
| E7 | DINOv3 as an image-only clustering view | 🟡 after the licence read, through the clustering plan's RQ-3b | no text tower (WB2 B4), so never a search space |
| E8 | LLM-based embedders (Qwen3-VL-Embedding 2B, Nemotron 3.5 Embed 1B) | 🟡 reference arms only | bound the gain from size; 4-16 GB of weights; no TensorRT; licence per card |
| E9 | jina-clip-v2 | ❌ | non-commercial licence on the card |
| E10 | A text-only small embedder (Qwen3-Embedding 0.6B or EmbeddingGemma v1) for OCR lines | 🟡 EP-4 stage B arm, same bars as E4 | a cheaper text-only option if the #347 candidate's licence or bf16 constraint blocks it |

### 4.3 Search and serving

| ID | Design | Verdict | Why |
|---|---|---|---|
| S1 | Query-vector cache in Valkey keyed by `(space_key, normalized text)`, TTL, size-capped | 🟡 EP-6 | the per-worker LRU misses across 32 workers and N replicas (section 2.3); Valkey is already required (D-69); a cache is not state, so statelessness holds |
| S2 | Text tower on the GPU (TensorRT plan beside the CPU onnxruntime model, same tensor names; selected per deployment) | 🟡 EP-6 | the encode is 53 ms of a 70 ms search; owner wants CPU or GPU selectable |
| S3 | Delete the in-process `torch` and `onnx` text backends | ✅ (J3) | D-69: Triton is a hard dependency, so the "search works with Triton down" fallback is dead code, and the in-process ONNX path is the 84 GiB trap |
| S4 | Hybrid OCR search: BM25 over item text plus a text-embedding kNN fused with RRF in an OpenSearch search pipeline | 🟡 EP-5 | standard documented pattern; only if placement A passes |
| S5 | Fusion across spaces (PE plus an optional space) by RRF after retrieval, never in one space | ✅ as the only allowed fusion | same rule as the #347 eval plan 2.3 |
| S6 | `group_by=image` by OpenSearch collapse on `image_id` with over-fetch (#436) | ✅ as designed in [10_v080_platform_breadth.md](10_v080_platform_breadth.md) F2, plus a `space` parameter | documented collapse; spike SP-1 there |
| S7 | `target=frame` (#435) as a separate route over the images index's frame space | 🟡 EP-2 decides | if the PE frame vector is within 0.02 recall@10 of MobileCLIP whole-image search, build it on the frame space; else close #435 (clustering plan RQ-10 bar, kept) |
| S8 | Lucene engine, `m` 16, `ef_construction` 128 for every new space | ✅ default, 🟡 at 1M (EP-3) | measured best filtered recall and half the disk |
| S9 | Quantized storage (Lucene 4x/8x SQ; faiss `on_disk` 16x/32x with rescoring; OpenSearch 3.9 `half_float`) | 🟡 EP-3 | 4.1 KiB/vector is fine to 1M (30 GB per field); quantize only after a real-vector recall gate (storage note rule 5) |
| S10 | Matryoshka cuts for an optional space | 🟡 EP-4 S6 | 256-d is a quarter of the bytes; the cut is part of the space key |
| S11 | GPU index build (OpenSearch remote index build with cuVS) | ❌ now | build time is not a measured bottleneck below 1M; revisit if EP-3 shows merges over the 1M budget |
| S12 | Separate shards or time partitions per project index at 1M | 🟡 EP-3 | the 50 GB shard guidance is reached only on faiss (61 GB); Lucene at 30 GB fits one shard |

## 5. Seams the v0.7.0 rebuild must include

These are the only product changes this plan asks of v0.7.0. They make a later encoder change a
data operation instead of a rewrite, and they replace a hidden assumption (one encoder, found by a
constant, identified by its dimension) with a stored fact. Each seam names its owning plan file and
lane. Section 11 lists the exact edits to those files.

### 5.1 Names and vocabulary (`src/domain/embedding_spaces.py`, pure, new)

| Term | Meaning | Type and rule |
|---|---|---|
| model key | a generic encoder name, never a vendor or private name | `^[a-z][a-z0-9_]{1,39}$`, for example `pe_core_l14_336`, `mobileclip2_s2`, `alt_text_a` |
| model version | an immutable encoder build: weights revision, input preprocessing, precision, export | integer >= 1 per model key |
| space | one vector field in one project index, written by one model version for one target at one dimension | `space_key = f'{model_key}_v{version}'` plus `_d{dim}` when the stored dim is a Matryoshka cut; regex `^[a-z][a-z0-9_]{2,47}$` |
| target | what the vectors describe | `item` (crop), `box` (region box, nested), `frame` (whole image), `item_text` (OCR and VLM text of an item) |
| role | `primary` (searched by default for its target), `optional` (searched only when named), `candidate` (being built or evaluated, never searched by default) | one primary per (project, target) |
| status | `pending`, `building`, `ready`, `retiring`, `retired`, `failed` | transitions in 5.5 |
| physical names | `vector_field` (`pe_embedding` for the upgraded spaces, `vec_<space_key>` for new ones), `state_field` (`embedding_state`, or `vec_<space_key>_state`), `nested_path` for `box` (`region_box_embeddings`, or `boxvec_<space_key>`) | stored in the registry, never computed by a reader |

New spaces for `box` get their own nested sibling (`boxvec_<space_key>` with `box_id`, `bbox_norm`,
`embedding`) rather than a second leaf inside `region_box_embeddings`, so one space's writer never
rewrites another space's nested list (the same reason `region_box_embeddings` is a sibling of
`region_boxes` today, `bodies_core.py:212-218`). Cost: one more nested Lucene document per box and
space; only optional spaces that target boxes pay it (J12).

### 5.2 PostgreSQL tables (owner: [02a](02a_postgres_schema.md), new section "Embedding registry"; created in `0001_control_plane` by lane 1A of [03a](03a_stage1_foundation.md))

```sql
CREATE TABLE embedding_model_versions (           -- deployment-wide catalog; no project_id, no RLS
  id                   uuid PRIMARY KEY DEFAULT uuidv7(),
  model_key            text NOT NULL CHECK (model_key ~ '^[a-z][a-z0-9_]{1,39}$'),
  version              integer NOT NULL CHECK (version >= 1),
  modalities           text[] NOT NULL CHECK (modalities <@ ARRAY['image','text'] AND cardinality(modalities) >= 1),
  native_dim           integer NOT NULL CHECK (native_dim BETWEEN 8 AND 4096),
  matryoshka_dims      integer[] NOT NULL DEFAULT '{}',          -- allowed cuts, each < native_dim
  distance             text NOT NULL CHECK (distance IN ('cosine')),
  input_spec           jsonb NOT NULL,                           -- pydantic EncoderInputSpec (resize and crop rule, size, mean/std, context length, query/document prefixes, image token budget)
  input_spec_sha256    text NOT NULL,
  weights_ref          text NOT NULL,                            -- source and pinned revision, for example 'hf:<repo>@<commit>'
  weights_sha256       text NOT NULL,
  precision            text NOT NULL CHECK (precision IN ('fp32','bf16','fp16','int8')),
  triton_image_model   text NULL,
  triton_text_model    text NULL,
  engine_parity_min_cos real NULL CHECK (engine_parity_min_cos BETWEEN 0 AND 1),
  licence              text NOT NULL,                            -- SPDX id or 'see ATTRIBUTION.md'
  status               text NOT NULL CHECK (status IN ('available','deprecated','retired')),
  created_at           timestamptz NOT NULL DEFAULT now(),
  created_by           text NOT NULL,
  UNIQUE (model_key, version),
  CHECK ((triton_image_model IS NOT NULL) = ('image' = ANY(modalities))),
  CHECK ((triton_text_model  IS NOT NULL) = ('text'  = ANY(modalities)))
);
-- trigger embedding_model_versions_immutable: BEFORE UPDATE raises unless only status changes

CREATE TABLE embedding_spaces (
  id                uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id        uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  space_key         text NOT NULL CHECK (space_key ~ '^[a-z][a-z0-9_]{2,47}$'),
  model_version_id  uuid NOT NULL REFERENCES embedding_model_versions(id) ON DELETE RESTRICT,
  target            text NOT NULL CHECK (target IN ('item','box','frame','item_text')),
  dim               integer NOT NULL CHECK (dim BETWEEN 8 AND 4096),
  role              text NOT NULL CHECK (role IN ('primary','optional','candidate')),
  status            text NOT NULL CHECK (status IN ('pending','building','ready','retiring','retired','failed')),
  index_role        text NOT NULL CHECK (index_role IN ('images','items')),
  vector_field      text NOT NULL,
  state_field       text NULL,                -- NULL only for an upgraded frame space (images carry no state field in 0.6.0)
  nested_path       text NULL,
  knn_method        jsonb NOT NULL,           -- pydantic KnnMethodSpec {engine, space_type, m, ef_construction, mode, compression_level, data_type}
  revision          bigint NOT NULL DEFAULT 1,
  built_by_job_id   uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  ready_at          timestamptz NULL,
  retired_at        timestamptz NULL,
  created_at        timestamptz NOT NULL DEFAULT now(),
  created_by        text NOT NULL,
  UNIQUE (project_id, target, space_key),
  UNIQUE (project_id, index_role, vector_field),
  CHECK ((target = 'box') = (nested_path IS NOT NULL)),
  CHECK ((target = 'frame') = (index_role = 'images')),
  CHECK (role <> 'primary' OR status = 'ready')
);
CREATE UNIQUE INDEX uq_embedding_spaces_one_primary
  ON embedding_spaces (project_id, target) WHERE role = 'primary';

CREATE TABLE embedding_reindex_runs (       -- the lasting record of one build; the job row carries status and progress
  id                uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id        uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  space_id          uuid NOT NULL REFERENCES embedding_spaces(id) ON DELETE CASCADE,
  job_id            uuid NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE RESTRICT,
  mode              text NOT NULL CHECK (mode IN ('build','backfill','clear')),
  scope             jsonb NOT NULL DEFAULT '{}',   -- optional item filter (pydantic ReindexScope)
  docs_total        bigint NULL,
  docs_done         bigint NOT NULL DEFAULT 0,
  docs_failed       bigint NOT NULL DEFAULT 0,
  docs_skipped      bigint NOT NULL DEFAULT 0,     -- not selected by the ingest policy, or no source pixels
  bytes_estimate    bigint NULL,                   -- shown before the start: docs x dim x bytes per value x index factor
  started_at        timestamptz NULL,
  finished_at       timestamptz NULL
);

CREATE TABLE embedding_evaluation_runs (    -- in-product evaluations only; research numbers live in docs/research
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id         uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id             uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  baseline_space_id  uuid NOT NULL REFERENCES embedding_spaces(id) ON DELETE CASCADE,
  candidate_space_id uuid NOT NULL REFERENCES embedding_spaces(id) ON DELETE CASCADE,
  protocol           text NOT NULL CHECK (protocol IN ('label_consistency_v1','query_set_v1')),
  sample_spec        jsonb NOT NULL,                -- seed, strata, n, holdout excluded
  metrics            jsonb NOT NULL,                -- pydantic SpaceEvalMetrics: recall@k by class, MRR, intervals, per-stratum
  verdict            text NOT NULL CHECK (verdict IN ('better','tie','worse','inconclusive')),
  created_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,
  CHECK (baseline_space_id <> candidate_space_id)
);
```

RLS: `embedding_spaces`, `embedding_reindex_runs` and `embedding_evaluation_runs` get the standard
project policy of 02a section 1.4. `embedding_model_versions` is deployment data, written only by the
`deployment` scope (the catalog sync and the CLI), read by all.

Rows written at creation and upgrade:

| When | Rows | Owner |
|---|---|---|
| `db-migrate` (deployment) | `embedding_model_versions` for `pe_core_l14_336` v1 and `mobileclip2_s2` v1 from the shipped catalog `examples/embedding/catalog.tsv` (new, same reader shape as `examples/vlm/catalog.tsv`), weights refs and sha256 from the export scripts | lane ES-A |
| project create (the create transaction of 02a section 2) | three `pe_core_l14_336_v1` spaces, all `primary` and `ready`: `item` (`pe_embedding`, state `embedding_state`), `box` (nested `region_box_embeddings`, leaf `embedding`; the per-box state stays derived from the box list as today), `frame` (images `pe_embedding`, no state field) | [03b](03b_stage2_control_data.md) lane 2A |
| the one-step 0.6.0 upgrade | the same rows for every loaded project, `created_by = 'upgrade'`, `knn_method` read from each live index mapping (faiss for old projects, Lucene for new: the registry records what is there) | 03d lane 2E |

The `box` space's `vector_field` for the upgraded PE space is the nested leaf `embedding` under
`nested_path = 'region_box_embeddings'`; the unique constraint uses `vector_field` as written
(`region_box_embeddings.embedding`).

### 5.3 OpenSearch field naming and versioning (owner: lane ES-A for the builders; [opensearch_schema_design.md](../../opensearch_schema_design.md) documents it in lane ES-D)

| Rule | Detail |
|---|---|
| Field names | new spaces: `vec_<space_key>` (`knn_vector`), `vec_<space_key>_state` (keyword, values of `embedding_state`), box spaces `boxvec_<space_key>` (nested: `box_id` keyword, `bbox_norm` float not indexed, `embedding` knn_vector) |
| Creation | `put_mapping` adds the fields when the space moves `pending -> building` (fields are additive; `index.knn` is already true on both project indexes) |
| Method | from `embedding_spaces.knn_method`, built by one function `knn_field_body(spec, dim)` that replaces `_knn_field` (`base.py:27`) |
| Version | the model version is in the key, so a new version is a new field; the old field's mapping stays after retirement (OpenSearch cannot drop a mapped field), its values are cleared by the `clear` mode of the reindex job, and `opensearch_schema_design.md` lists retired fields per project through the registry |
| Field budget | `index.mapping.total_fields.limit` (default 1000) is far away; at most 3 live spaces per (project, target) is a registry constraint checked in the service (fail closed with `embedding_space_limit`) |
| Read projection | every item read keeps excluding every vector field; the exclusion list is built from the registry (`item_list_source_excludes` gains the registry fields), never a hand list |
| Core indexes | the five `visual_search_*` indexes stay deployment-wide with the `mobileclip2_s2` v1 and ArcFace spaces recorded as constants in `src/clients/opensearch/names.py` plus a test that their dims equal the catalog rows; a registry for core indexes is out of scope (J10) |

### 5.4 Readers resolve their field (owner: lane ES-C)

- `ProjectContext` (`src/config/project_context.py`) gains `embedding_space(target: Target, key:
  str | None = None) -> EmbeddingSpace` (frozen dataclass: `space_key`, `vector_field`, `state_field`,
  `nested_path`, `dim`, `model_key`, `model_version`, `triton_text_model`, `triton_image_model`), loaded
  with the project record and refreshed on `projects.config_epoch` (the flip bumps it, 02a section 4).
- `ITEM_EMBEDDING_FIELD` and `DEFAULT_KNN_FIELD` are deleted; `src/domain/embedding_state.py` clauses
  take an `EmbeddingSpace`. A ratchet test `tests/curation/test_no_embedding_field_literals.py` fails
  on any `pe_embedding`, `region_box_embeddings` or `ITEM_EMBEDDING_FIELD` literal outside the registry
  modules, the upgrade loader and the mapping builders (the `test_embedding_state_consumers.py`
  pattern).
- The query builder (`build_knn_query`) takes the space and the query vector's `space_key`; a
  mismatch raises `EmbeddingSpaceMismatch` (500 in production, a test proves it fires).
- Combine: a source's vectors are copied only for spaces whose `space_key` exists in the target; the
  rest are dropped and their items marked `deferred` for that space; the preview warning
  `embedding_model_mismatch` compares space keys, not dimensions (replaces `warnings.py:42-60`).

### 5.5 The re-index Celery job (owner: [03d](03d_stage4_job_families.md), new lane ES-B; kind row in [02b](02b_jobs_locking_events.md) section 2.1)

| Item | Value |
|---|---|
| Kind | `embedding_reindex`, module `src/config/job_kinds/b2_embedding.py`, task `op.jobs.embedding_reindex`, queue `jobs`, `exclusive_group` `embedding_reindex` with `scope = space_key` (one build per space), `auto_resume` true, time limit 24 h (D-20a's long-job class), `max_attempts` 3 |
| Params (pydantic `EmbeddingReindexParams`) | `space_key`, `mode` (`build` all docs of the target, `backfill` only docs whose state field is missing or `deferred`/`failed`, `clear` remove the values after retirement), `batch_size` (default 64 images), optional `scope` |
| Steps | (1) `pending -> building` in one transaction with `put_mapping`; (2) iterate the index with PIT and `search_after` on `_id` (no scroll), fetching only ids, geometry and image references, never vectors; (3) decode each image once through the B5 decode path and encode every target of the batch on Triton; (4) write the new field and its state with the OCC partial-update helpers (`src/clients/occ*.py`); the lock rule does not apply to vectors, but a box whose `bbox_norm` changed since the read is skipped and left `deferred`; (5) fenced checkpoint `{pit_after, docs_done}` every batch (02b section 3); (6) at the end, `building -> ready` and the run row's counts |
| Progress | `jobs.progress = {done, total, stage: 'encode'|'write'|'verify', counters: {failed, skipped}}` every batch; one `job.progress` event per 2 % (02b section 6) |
| Cancel and pause | cooperative at batch boundaries (02b section 3.3); cancel leaves the space `building` with its partial field (re-startable as `backfill`) or, with `abandon=true`, `failed` and a `clear` run queued |
| Idempotency | a re-run writes the same vector to the same field; the `Idempotency-Key` header of D-58 on the start route |
| Verify | after the build, a sampled count `exists(vector_field)` equals `docs_done`; 200 random items re-encoded and compared (cosine >= 0.999) before `ready` |
| Dependent artefacts | clustering, the IVF centroid store and the near-duplicate groups read the primary space only; the flip queues the cluster refresh and the near-dup re-score for that space (the clustering plan records `space_key` on every cluster run, section 11) |

Flip (`promote`): one transaction `UPDATE embedding_spaces SET role = CASE id WHEN :new THEN
'primary' WHEN :old THEN 'optional' END, revision = revision + 1 WHERE project_id = :p AND id IN
(:new, :old) AND revision IN (...)`, the event row, the `config_epoch` bump; refused with 409
`embedding_space_not_ready` unless the candidate is `ready`, and with 409 `embedding_evaluation_missing`
unless an `embedding_evaluation_runs` row for the pair exists (the owner may waive with `force=true`,
J11). Retire: `optional -> retiring`, a `clear` run, then `retired`. The primary can never be
retired.

### 5.6 Config kinds

- **No new `config_documents` kind.** A space has identity, status, an owner job and a lifecycle,
  which 02 section 2 puts in relational tables, not in revisioned documents.
- **`project_settings.search` (new `jsonb NULL` column)**, pydantic `SearchSettings`: default `group_by`
  (`none`), the RRF `rank_constant` (60), the over-fetch factor for collapse (5), the default optional
  spaces to fuse (empty). Written with `expected_revision` like the other settings (02a section 4).
- **The deployment catalog** `examples/embedding/catalog.tsv` (model key, version, modalities, dims,
  Triton model names, weights ref, sha256, licence). A model whose Triton models are not installed is
  listed as `unavailable` and cannot back a space (fail closed). Optional encoders install with
  `./openprocessor models install --only embed-<model_key>`; nothing optional loads into Triton by
  default.

### 5.7 API and wire (owner: lane ES-B for the routes, ES-C for the search and item wire; contracts regenerated in each)

Proposed routes (they do not exist yet):

| Method and path | What | Response |
|---|---|---|
| GET `curation/embedding/models` | the catalog with availability | `EmbeddingModelListResponse` |
| GET `curation/projects/{project}/embedding/spaces` | the project's spaces | `EmbeddingSpaceListResponse` (`space_key`, `target`, `model_key`, `model_version`, `dim`, `role`, `status`, `ready_at`, `docs_with_vector`) |
| POST `curation/projects/{project}/embedding/spaces` | register a space (`model_key`, `model_version`, `target`, optional `dim`) and start its `build` job | 202 with `job_id` and `bytes_estimate`; 409 `embedding_space_exists`, 422 `embedding_model_unavailable` |
| POST `curation/projects/{project}/embedding/spaces/{space_key}/promote` | the fenced flip | 200 with both spaces; 409 as in 5.5 |
| POST `curation/projects/{project}/embedding/spaces/{space_key}/retire` | retire an optional space and queue `clear` | 202 |
| POST `curation/projects/{project}/embedding/spaces/{space_key}/evaluate` | start an in-product evaluation against the primary | 202, row in `embedding_evaluation_runs` |
| GET `curation/projects/{project}/embedding/reindex-runs/{run_id}` | the run with its job's progress | `EmbeddingReindexRunResponse` |

Cancel and pause use the generic job routes of [03c](03c_stage3_job_runtime.md) (3A-9).

Search wire (the three existing routes `/curation/projects/{project}/search/text`, `/search/similar`,
`/search/image`): a `space` query parameter (default: the target's primary), echoed in the response as
`space`; `fuse` (a list of optional space keys, RRF after retrieval, design S5); items keep `score` and gain
`score_space`. Item wire: `embedding_state` becomes `embedding_states: {space_key: state}` built from
the registry (one serializer, `src/services/curation/wire.py`). `group_by` from v0.8.0 F2 composes with
`space`.

## 6. Prototype protocol

### 6.1 Code location, environment and data handling

- **Code: tracked, `scripts/research/embedding_search/`**, with exact `==` pins in
  `scripts/research/embedding_search/requirements.txt` and a `pip freeze` captured in every result
  JSON. Justification: the research rules require a reproduce command from a clean checkout
  ([docs/research/CLAUDE.md](../../research/CLAUDE.md)); the code is small and reviewed; the naming scan
  covers it, which is what keeps vendor names out of arm keys. A gitignored scratch harness (the eval
  plan's `$HOME/eval_work/`) makes every number unreproducible and was rejected for that reason.
  Data, vectors, engines and raw logs are never tracked.
- **Layout** (it absorbs the `embedding_eval/` package the #347 plan proposed):

```text
scripts/research/embedding_search/
  arms.py         arm registry: key, loader, dtype (refuses fp16 for bf16-only arms), device, input path
                  (square | processor), prefixes per role (query | document | none), native dim, cuts
  data.py         pinned manifests: COCO 2k pin, COCO 5k Karpathy test, Open Images LN sample, TextCaps/TextOCR,
                  curation crop sets, VIBE and big-ann slices, set P (path from $SETP_ROOT, never in a file)
  encode.py       decode once (PIL, exif_transpose, RGB), feed every arm from that frame; sharded, resumable
                  float32 .npy with metadata (arm, revision, dim, cut, normalization)
  score_xmodal.py text-to-image, image-to-text, image-to-image, RRF fusion
  score_text.py   OCR and caption retrieval, lexical baselines (OpenSearch BM25, the product token rule)
  score_cluster.py partitions, then scripts/bench/cluster_eval.py
  index_bench.py  drives scripts/bench/vectordb/ adapters for EP-3 (no new engine code)
  latency.py      one process per (arm, device, batch), fresh CUDA context; text-encode serving rows
  gates.py        gates A0-A6
  report.py       tables from result JSONs, never typed by hand
  run_gpu.sh      wrapper: the GPU lock, telemetry, refusal when a foreign process holds the card
tests/research/test_embedding_search.py
```

- **Venv:** a local research venv outside the product requirements, created with
  `python3 -m venv "$SCRATCH/venvs/embsearch"` and `"$SCRATCH/venvs/embsearch/bin/pip" install -r
  scripts/research/embedding_search/requirements.txt` (CUDA torch from the torch index, recorded). No
  Docker builds. The only containers are stock images pulled by digest: OpenSearch for EP-3 and EP-5
  under `docker compose -p op-rs-emb -f docker/bench/vectordb/compose.yml`, never the default project.
- **Libraries before code:** `sentence-transformers` and `transformers` (loading, prefixes, Matryoshka),
  `open_clip_torch` and `timm` (CLIP-family arms), FAISS `IndexFlatIP` (exact ground truth), `mteb`
  (anchors), OpenSearch's own BM25 (the lexical baseline a user would get; the `rank-bm25` package of
  the eval plan is dropped), the repo's tested metric code (`retrieval_eval_metrics.bootstrap_ci`,
  `standard_benchmarks.metrics.paired_bootstrap`, `cluster_eval.py`), NVML for memory, Polygraphy,
  `trtexec` and ModelOpt for stage D. No new recall, nDCG or bootstrap code.
- **Names:** arm keys are generic (`emb_t`, `emb_tv` for the #347 candidate's text and text+vision
  loads, `pe_core_l`, `pe_core_g`, `mobileclip2_s2`, `mobileclip2_s4`, `siglip2_so400m`, `txt_small_a`,
  `ref_vl_2b`, `ref_nv_1b`); every non-baseline Hugging Face id comes from `EMB_SEARCH_<KEY>_HF_ID` or
  `--hf-id`, so no tracked file spells a banned name.

### 6.2 Datasets and sampling

| Id | Set | Sampling | Used by | Licence and publication |
|---|---|---|---|---|
| D1 | COCO val2017 2,000 pin, 5 captions | `docs/benchmarks/retrieval_baseline/sample_ids.json` (seed 20261028); **not** `coco_bench_2000.json`, which mixes train and val (the clustering plan's 4.1 premise, corrected) | EP-1, EP-2, EP-4 | captions CC BY 4.0, images Flickr terms; aggregates published |
| D1k | COCO Karpathy 5k test | the standard split file, sha256 recorded | gate A6 (PE card anchor), EP-4 S4 | as D1 |
| D2 | Open Images V7 validation, 5,000 with Localized Narratives | seeded; new pin `scripts/datasets/manifests/oi_ln_5000.json` written by extending an existing fetcher | EP-4 S3/S4 | CC BY 4.0 annotations, per-image licence recorded |
| D3 | TextCaps + TextOCR on Open Images train (disjoint from D2) | new pin `textcaps_textocr_<n>.json` | EP-4 S2, EP-5 | CC BY 4.0 annotations |
| D4 | curation crop sets `cluster_eval_fresh` (acceptance) and `cluster_eval_3000` (regression) | existing manifests | EP-4 S5 | published as before |
| D5 | CUB-200-2011, SOP | WB2 splits | EP-4 image-to-image rows | local only, aggregates as WB2 |
| D6 | real PE vectors at 1M: every crop and frame of COCO train2017 plus val2017 under policy `all` (about 0.8M at 6.5 per image) plus Open Images V7 validation crops to pass 1,000,000 | no jitter, no synthetic padding | EP-3 | aggregates published |
| D7 | VIBE `laion-clip-512-normalized` (OOD text queries) and `imagenet-clip-512-normalized` | the published files with their ground truth | EP-3 public rung | public |
| D8 | big-ann NeurIPS'23 filtered, a 1M slice with tag filters | seeded slice | EP-3 filter behaviour | public |
| D9 | MTEB tasks named on each candidate's card | one small retrieval or STS task | gate A4 | public |
| P | set P: the private high-resolution photo set, its crops and frames (about 1.2M crops exist in the measured project) | read-only export of stored vectors plus a seeded 5,000-image re-encode sample | EP-2, EP-3, EP-4 confirmation rows | **private**: results stay out of the public tree (6.8) |

Quality on set P has no human labels: queries are written from the project's class vocabulary
(200 seeded phrases) and the top-10 of each arm are judged by the visual judge only on the kinds
that passed calibration in #406, plus a 5 % blind human re-judge by the owner. No quality claim is
made on synthetic or jittered vectors (owner rule).

### 6.3 Metrics

| Family | Metrics | Code |
|---|---|---|
| Retrieval quality | recall@1/5/10, MRR@10, nDCG@10 (first-relevant), 95 % bootstrap over queries (1,000 resamples); paired bootstrap for deltas | `retrieval_eval_metrics`, `standard_benchmarks.metrics` |
| Image-to-image | precision@10 with the category-Jaccard proxy (D1); Recall@K DML protocol (D5) | same |
| Clustering view | BCubed precision and F at the calibrated cut, pairwise, AMI, ARI between partitions, per stratum | `cluster_eval.py` |
| Index | recall@10 and recall@100 against exact filtered ground truth; "returned fewer than k" rate; p50/p95/p99 at concurrency 1, QPS at 8 and 32; time to searchable; disk KiB/vector after settle; native and page-cache memory | `scripts/bench/vectordb/` |
| Encoder cost | images/s and texts/s by batch, ms per query text (p50/p95/p99), peak VRAM, vector bytes | `latency.py` |
| Product path | served p50/p95/p99 and the `Server-Timing` stages at 100k and 1M | `retrieval_eval.py latency` |

### 6.4 Repetition and noise

Quality runs are deterministic: one run plus bootstrap intervals; a re-run only to confirm a decisive
row. Latency and index rows: 3 repetitions, median and min-max; latency uses at least 128 discarded
warm-ups and 600 measured distinct queries (the #414 protocol). A row taken at a 1-minute load above 16
is labelled noisy and repeated before it may decide anything. Every result JSON carries git sha, arm,
model revision, dtype, device, dataset pin hash, seed, `nvidia-smi` snapshot, load average start and
end, wall time, peak VRAM, and for CPU rows the thread count and core list.

### 6.5 GPU etiquette

- GPUs 0 and 2 only (RTX A6000); never GPU 1. `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
  `CUDA_VISIBLE_DEVICES` set explicitly.
- **One GPU job at a time per card**, under the program's per-card lock (00b X-14):
  `flock "$SCRATCH/locks/gpu<N>.lock" <command>` inside `run_gpu.sh`, `<N>` the chosen card; the wrapper refuses to start
  when a process outside the lock holds more than 2 GiB on the chosen card.
- Before a window: `uptime`, `nvidia-smi`, tell the peer sessions what will run and for how long; pause
  at once when a peer asks. Run under `nice -n 10`. VLM captioning (EP-4 S3) runs before any latency
  measurement and the VLM is stopped afterwards.
- Research windows are outside the program's three GPU windows (MQ-9): RW-E1 (EP-1, EP-2, EP-4 stage A
  and the S4 screen; about 3 h), RW-E2 (EP-3 encodes of D6 and the index arms; about 6 h GPU, CPU-heavy
  index runs after), RW-E3 (EP-4 stage C, EP-6 GPU text engine; about 6 h). They share vectors with the
  clustering plan's RW-1 and RW-2 where the arms coincide (encode once).

### 6.6 The prototypes

| ID | Question | Method and arms | Data | Bars (registered in EP-0 before any read) | Stop rule | Decides |
|---|---|---|---|---|---|---|
| EP-0 | registration | the bars of this table and 6.7 committed in `docs/research/embedding_search_results.md` (new) | none | docs only | n/a | n/a |
| EP-1 | is the offline harness the product? | gate A5: offline MobileCLIP2-S2 on D1, first caption | D1 | recall@1 in [0.546, 0.588] and recall@10 in [0.872, 0.901] | no scenario is read until it passes; typical causes in order: EXIF, resize rule, normalization constants, self-match in the ranking | trust in every later row |
| EP-2 | PE whole-frame search against MobileCLIP and against PE crops (RQ-10, E-12; #435) | PE-Core-L frame vectors (centre square, the production rule) text-to-image and image-to-image; gate A6 first: PE on D1k within 1.0 point of the card's 57.1 | D1, D1k; set P 200 judged queries | frame recall@10 within 0.02 of MobileCLIP, or better | if A6 fails, fix the PE path before reading anything; if the bar fails, #435 closes as "whole-image search stays in the core API" | #435, the frame target, the default `target` of image search |
| EP-3 | index settings at 1M real vectors | arms: Lucene m16 efc128 (today), Lucene 8x and 4x SQ with rescoring, faiss `on_disk` 16x and 32x with rescoring, OpenSearch 3.9 `half_float` on Lucene (stock 3.9 image, own `-p`), Lucene m32 only if a filter band misses the bar; filters at 0.5, 3, 5, 15, 45 % by measured selectivity; k 10, 200, 2000 | D6 (public), D7 and D8 (public replication), set P vectors (private) | on the product filters: recall@10 >= 0.95 against exact; kNN p99 <= 60 ms at 1M at every band; served search p99 < 300 ms (the #414 bar); disk at 1M per field reported | a quantized arm that misses recall@10 0.95 at its best setting is dropped; if Lucene m16 meets every bar, nothing changes; a split to another store is reopened only if no OpenSearch arm meets the bars (the #221 trigger) | S8, S9, S12; the OpenSearch 3.9 bump (J4) |
| EP-4 | the #347 evaluation and the other encoder candidates | the #347 stages A-D (6.7), arms E4-E8, E10 | D1, D1k, D2, D3, D4, D5, D9; set P confirmation of any winner | the #347 placement bars (6.7) | the #347 stop rules (6.7) | #347; optional spaces; RQ-3b row for the clustering plan |
| EP-5 | hybrid OCR text search | BM25 (OpenSearch `match` on item text), the product token rule, a text-embedding kNN, and the `hybrid` query with RRF and with min-max normalization, in a throwaway OpenSearch | D3, D3 with PP-OCRv5 strings (1,000 images, read by the stack under `-p op-rs-emb-ocr`) | the hybrid arm beats the best single arm by non-overlapping intervals on recall@10, or adds >= 5 points in the union | run only if EP-4 placement A is Go or Conditional | S4 |
| EP-6 | text encode on the GPU and a shared query cache | PE text tower: Triton onnxruntime CPU (today) vs a TensorRT GPU plan with the same tensor names (fp16 parity >= 0.999 cosine to fp32 on 1,000 real captions); cache: replay a query log (COCO captions with a Zipf repeat pattern, and the set-P query log if the owner provides one) against per-worker LRU at 32 workers and against one Valkey cache | D1 captions; set P log (private) | GPU path: p50 <= 10 ms and p99 <= 25 ms at concurrency 8 with no VRAM above 1 GiB; cache: hit rate at least double the per-worker LRU | if the GPU plan fails parity, it is recorded as blocked and the CPU path stays; if the cache gains less than 2x hit rate, no cache | S1, S2 |
| EP-7 | the dual-field swap rehearsal (after the section 5 seams exist) | on a throwaway project of 100k real items: register MobileCLIP2-S2 as a candidate `item` space (a real second encoder with another dimension, already in the model repository), run `build`, measure items/s, disk growth, served p99 during the build, kill a worker mid-run and resume, flip, `clear`, settle | D6 subset, stack `-p op-rs-emb-swap` | resume re-encodes nothing finished; search p99 during the build < 300 ms; flip visible to every worker within one config poll; disk returns within 10 % of the start after `clear` plus merge | any failure is a product bug, fixed in the seam lane before v0.8.0 F9 | F9 readiness |

### 6.7 The #347 evaluation (absorbed from the deleted issue plan, unchanged except where marked)

**Stage A, gates** (each writes one result JSON; a failed gate stops the arm with its cause):

| Gate | What | Pass |
|---|---|---|
| A0 licence | the model card licence, the linked base-family licence text and the prohibited-use policy read in full (owner, or the agent reporting to the owner) | compatible with an optional, off-by-default integration in an AGPL repo with an ATTRIBUTION row; a mismatch is a no-go for shipping, not for measuring |
| A1 revision pin | Hugging Face commit sha of weights and processor; `transformers` and `sentence-transformers` versions | recorded |
| A2 precision | 200 real texts and 200 real images: bf16 against fp32 cosine | mean >= 0.999, min >= 0.99, no NaN; a deliberate fp16 run recorded as the expected failure |
| A3 one space | 270M text-only load against the 440M text+vision load on 1,000 texts | cosine >= 0.999; otherwise cross-modal rows use the 440M text path only |
| A4 loader sanity | one small MTEB retrieval or STS task named on the card, with the card's prompts | within 1.0 point of the vendor value, or "no anchor" recorded |
| A5 harness parity | = EP-1 | as EP-1 |
| A6 PE anchor (new) | = EP-2's gate | PE-Core-L COCO 5k text-to-image within 1.0 point of 57.1 |

**Stage B, text only (mostly CPU):** S2 OCR and item text: S2a meaning search (TextCaps caption
against the image's concatenated TextOCR strings), S2b noisy string (seeded perturbation of one OCR
line), S2c real OCR noise (PP-OCRv5 strings on 1,000 D3 images), S2d union of `emb_t` and lexical
top 10; arms `emb_t` at 768/256/128, `txt_small_a`, OpenSearch BM25, the product token rule
(`src/domain/item_text.py`), the PE text tower as a naive text-to-text arm. S3 captions: index A is
`emb_t` over captions (a held-out human caption, then the catalog VLM's caption with a recorded
prompt), index B is the image vectors (PE-Core-L, MobileCLIP2-S2, `emb_tv`); a different human
caption is the query; recall@k and top-10 overlap. S6 text half: every stage-B metric at 768, 512, 256,
128 with re-normalization, int8 at 256, and the on-disk size of a throwaway Lucene index. CPU text
latency in fp32 (the host CPU is AVX2-only, no bf16 or VNNI), 8 pinned threads, 5-20 token queries and
documents to 512 tokens, batch 1 and 8, p50/p95/p99 over 30 iterations after 3 warm-ups, 3 repetitions.

| Placement | Go | Conditional | No-go |
|---|---|---|---|
| A, OCR / item text | S2a recall@10 and S2b recall@1 above BM25 and the token rule with non-overlapping 95 % intervals, or the S2d union adds >= 5 recall points at 256-d; CPU p95 <= 50 ms for a 20-token query, or GPU batched at most the PE text cost | better only at 768-d, or only for noisy strings | no gain over lexical, or CPU p95 > 150 ms |
| B, captions | caption-index recall@10 >= direct image search and union recall up >= 5 points | equal but fully overlapping | worse and not complementary |
| Storage | 256-d loses <= 3 % relative on stage-B metrics; 128-d <= 10 % for text-only | | 256-d loses > 5 % |

**Stage C, text plus vision (RW-E3):** S4 text-to-image and image-to-text on D1 and D1k (5 captions
are the relevants), image-to-image on D1 (proxy) and D5; arms PE-Core-L, MobileCLIP2-S2, `emb_tv` at
four cuts and two token budgets (140, 280), `siglip2_so400m` and `mobileclip2_s4` on the S4 screen
only, RRF fusion of PE and each candidate after retrieval. The **S4 screen** (used by the stop rule):
D1 text-to-image recall@5 of `emb_tv` at 768-d against PE-Core-L and MobileCLIP2-S2, plus fusion. S5
clustering view on D4 with the production IVF (K = min(512, n // 6)) and average linkage at a
dev-calibrated cut, scored per stratum, ARI against the PE partition; this is also the RQ-3b row of the
clustering plan, with its replacement bar marked "not applicable by owner rule". S1 speed in bf16,
batches 1 to 32, text lengths 16 to 512, image budgets 70 to 560, preprocessing timed apart; budget for
an optional embedder at most 2 GiB (text) and 4 GiB (vision) of VRAM (J9). S6 image half: as stage B
plus kNN p50/p95/p99 on a 100k pool in the throwaway OpenSearch. Reference arms `ref_vl_2b` and
`ref_nv_1b` on the S4 screen, S2a and S5 only, one working day each, then "not run" with the reason.

| Placement | Go | Conditional | No-go |
|---|---|---|---|
| C, second image view | S4 text-to-image recall@5 not below PE-Core-L at >= 256-d, image-to-image not below PE-Core-L, extra ingest cost inside the measured headroom | wins only in fusion (>= 3 points recall@10) | below PE-Core-L and no fusion gain |
| D, clustering view | partition scores at least equal, wins at least one stratum PE loses, ARI < 0.8 | equal everywhere | worse in every stratum |
| Hard gates | bf16 parity, no NaN in 100k encodes, licence (A0), a serving path on Triton or onnxruntime without an unmaintained fork, VRAM next to the current models on one card | | |

**Stage D, TensorRT and shipping, only for a component that passed:** the eval plan's ladder
(embedding_model_eval_plan.md 5.4-5.5): fp32 reference and bf16, fp16 built once to document the
failure, INT8 only if bf16 passes and speed matters; FP8 and NVFP4 are `skipped: hardware` on sm_86;
built in the serving Triton image, `trtexec --version` recorded; one working day per tier, then
`blocked`. The engine replaces onnxruntime only if >= 1.5x faster at the same batch or half the VRAM.
**Never call `trt_utils.bake_fp16_onnx` for a bf16-only model.**

**Stop rules:** licence incompatible: measure only if the owner still wants numbers, nothing ships;
A2 or A5 failing: fix or stop; after stage B, A and B both no-go **and** the S4 screen loses to
PE-Core-L and MobileCLIP2-S2 with no fusion gain: #347 is no-go, close with the evidence, no stage C
remainder, no reference arms, no TensorRT; after stage C, no placement Go or Conditional: same;
Conditional only: report, one follow-up only if the owner asks; stage D past its time box: serve
through onnxruntime if within the S1 budget, else the placement waits with the reason.

**Failure register (from the issue plan, kept):** fp16 by accident (WB2 wrapper autocast, a stray
`torch.autocast`, an fp16 ONNX export, the fp16 baker); missing or swapped task prefixes; Matryoshka
without re-normalization; query and gallery at different cuts (the metadata check refuses it); 270M and
440M text vectors that differ; token budget changing quality and cost (report both); EXIF orientation
(passes on COCO, fails on Open Images); captions or OCR strings leaking into the repo (ids and
aggregates only); the same caption as index document and query; D3 overlapping D2; pre-training
contamination (comparisons are relative); BM25 analyzer differing from the product token rule (report
both lexical arms); ties in exact kNN (break by id); CPU latency on a loaded host; vLLM older than the
sm_86 startup fix; a throwaway OpenSearch under the default project name; GPU 1; bulk-deleting raw
data; a per-worker CPU model in production (one Triton CPU instance only); "optional" must mean absent
when off (no plan loaded, no VRAM, no field, no wire field); licence drift between the read and the
release (ATTRIBUTION cites the revision read).

### 6.8 Recording results

- Public numbers: `docs/research/embedding_search_results.md` (new; registration first, results
  appended by `report.py`, one verdict line per placement and per EP), backing JSONs in
  `docs/benchmarks/embedding_search/` with `SHA256SUMS`, never pruned. Pointers: section 10 of
  [embedding_model_eval_plan.md](../../research/embedding_model_eval_plan.md), a row in
  `docs/research/README.md` and in the research table of [README.md](../README.md), the RQ-3b, E-4, RQ-10
  and E-12 rows of [09b_research_clustering.md](09b_research_clustering.md), an appended PE frame row
  in [retrieval_baseline.md](../../research/retrieval_baseline.md) (append, never rewrite), the 1M index
  rows in [PERFORMANCE.md](../../PERFORMANCE.md) (append; keep every existing number), comments on #347,
  #365 and #435.
- Set-P results: raw JSON in `artifacts_local/embedding_search/set_p/` (gitignored) and the
  maintainers' private ledger; the public note says only that a set-P confirmation ran and whether it
  agreed, with no numbers, folder names or paths (owner decision V1 of 2026-10-10).
- Raw runs, logs and GPU samples in `artifacts_local/embedding_search/`; never bulk-deleted.

## 7. Production design (the target after the research)

1. **One registry, many spaces.** PE-Core-L stays the primary space for items, boxes and frames;
   MobileCLIP2-S2 stays the core inference API's encoder. An optional space exists in a project only
   after an owner action (POST spaces) and only for an encoder in the catalog whose Triton models are
   installed. Off by default means: no catalog model loaded, no field, no wire field.
2. **Ingest writes every live space of each target in the same pass**, decoding each image once (B5)
   and encoding on Triton; the ingest policy's selection applies per space; a space in `building` gets
   new items from ingest and old ones from its job, so the job's `backfill` mode is the final sweep.
3. **Search names its space.** Default primary; `fuse` adds optional spaces by RRF after retrieval;
   hybrid OCR search (if EP-5 passes) runs as an OpenSearch `hybrid` query in a search pipeline whose
   name is per deployment. The query text is encoded on Triton (CPU or GPU plan per deployment),
   through a Valkey cache if EP-6 passes; the in-process backends are gone.
4. **Swap = build, evaluate, flip, retire.** The in-product evaluation (`label_consistency_v1`):
   for a seeded sample of class-validated items outside the holdout, recall@10 of same-class neighbours
   in each space with bootstrap intervals, per class; shown before the flip. Clustering, centroids and
   near-duplicate groups follow the primary space; the flip queues their refresh.
5. **Index settings:** Lucene HNSW, m 16, ef_construction 128 per space unless EP-3 says otherwise;
   quantization only through a new space (settings are part of the space).
6. **Encoders change only by a new model version.** A rebuilt engine that fails the parity gate is a
   new version; B3's GPU gate G7 (embeddings) records the parity number into the catalog.
7. **Combine and import:** vectors travel only between equal space keys; everything else is
   `deferred` and backfilled by the target's job.

## 8. Landing lanes

### 8.1 v0.7.0 (the seams; one lane per worktree, disjoint files, targeted tests only)

| Lane | Wave | Branch | Owns | Red tests (written first) |
|---|---|---|---|---|
| ES-A registry core | W2.1 (tables inside lane 1A's `0001_control_plane`) and W2.2 (code, beside 2A) | `b2/embedding-registry` | `src/domain/embedding_spaces.py`, `src/db/repositories/embedding_spaces.py` (path per 03a's repository layout), `src/services/embedding_spaces/{catalog,spaces,flip}.py`, `examples/embedding/catalog.tsv`, the `knn_field_body` builder (replaces `_knn_field` in `src/clients/curation_opensearch/base.py`), `tests/embedding_spaces/` | a second primary for one target is a unique violation; promote of a `building` space answers 409; a catalog row whose Triton model is missing is `unavailable`; the model-version trigger refuses an update of `input_spec` |
| (in 2A, 2E) default and upgrade rows | W2.2 | `b2/projects-pg`, `b2/upgrade-060` | 3 rows in the create transaction; rows per loaded project in the upgrade, `knn_method` read from the live mapping | project create leaves three `primary` `ready` spaces; an upgraded faiss project records `engine: faiss` |
| ES-B re-index job and routes | W2.3 | `b2/embedding-reindex` | `src/config/job_kinds/b2_embedding.py`, its task routes, `src/tasks/embedding_reindex.py`, `src/routers/curation/embedding_spaces.py` and `_embedding_space_models.py`, contracts | kill mid-run and resume: no finished doc re-encoded (fake Triton counts calls); cancel at a batch boundary leaves `building` and a valid checkpoint; a moved box is skipped as `deferred`; a second `build` for the same space is the unique-index 409 |
| ES-C readers to the registry | W2.4, parallel with 6A and 6B (the coordinator checks disjointness at the cut) | `b2/embedding-readers` | every file of `rg -l "pe_embedding|ITEM_EMBEDDING_FIELD|DEFAULT_KNN_FIELD" src scripts` not owned by 6A or 6B (about 40), `src/domain/embedding_state.py`, `src/services/curation/wire.py` (`embedding_states`), the search `space` parameter, `tests/curation/test_no_embedding_field_literals.py` | the ratchet fails on the constant today; a query vector of another space raises; combine copies only equal space keys; item wire shows `embedding_states` |
| ES-D docs | W2.4, after ES-C | `b2/docs` (lane 6D) | `docs/opensearch_schema_design.md` (spaces and naming), `docs/ARCHITECTURE.md` (the registry), `docs/CURATION.md` (spaces, swap) | docs check |
| ES-E delete the in-process text backends (D-69 consequence) | W4, inside lane 4B (worker knob and statelessness) | `b4/worker-knob` | `src/clients/pe_text_backends.py` (keep only Triton), `src/clients/pe_encoder.py`, `src/config/retired_env.py` (`OP_PE_TEXT_BACKEND`, `OP_PE_TEXT_ONNX_PATH` as `removed:`), `env.template`, the 503 text of `src/routers/curation/search.py:116` | a process with Triton's text model not ready refuses to serve search with an error naming it (no torch import anywhere in the API image path); the retired variables fail at startup |

Each lane: one agent in its own worktree, Sonnet implements, reviews by Sonnet; briefs forbid
`pgrep -f`, `pkill -f`, sleep and until loops, `git stash`, `--no-verify`, pushing and merging, and
require reading the cited docs (OpenSearch put mapping and knn-vector, PostgreSQL partial unique
indexes, Celery task routing, 02b) before coding and citing them in the commit.

### 8.2 After the v0.7.0 tag (research and the swap)

| Order | Work | Window | Output |
|---|---|---|---|
| 1 | EP-0 registration (docs only; may land any time, also before the tag) | none | `docs/research/embedding_search_results.md` |
| 2 | harness and tests (6.1), D2/D3 pins | CPU | lane tests green |
| 3 | EP-1, EP-2, EP-4 stage A, stage B (CPU), the S4 screen | RW-E1 | #435 decision; #347 stop-rule read |
| 4 | EP-3 (encodes of D6 on GPU, index arms on CPU) and EP-6 | RW-E2 | index settings; text-encode decision |
| 5 | EP-4 stage C, reference arms; EP-5 if placement A passed | RW-E3 | #347 verdict per placement |
| 6 | EP-7 rehearsal on the shipped seams | CPU plus one GPU hour | F9 readiness |
| 7 | v0.8.0 F9 reduced to: register the winning model version, build, evaluate, flip (or #365 parked), and F2 `group_by` with `space` | v0.8.0 window V-EM | #365 closes or is parked |
| 8 | an optional-space feature lane per passing placement (catalog row, Triton models, ATTRIBUTION, docs) | v0.8.0 | the placement ships off by default |

## 9. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| The seams grow B2 by three lanes | B2 slips | ES-A and ES-B add new files only; ES-C is mechanical and ratcheted; J1 lets the owner defer ES-B to v0.8.0 while keeping ES-A and ES-C |
| ES-C touches many files other lanes edit | merge conflicts | cut from the frozen W2.3 head; coordinator lists 6A/6B files first; the ratchet test proves completeness |
| Upgrade registers a wrong `knn_method` | a later space built with the wrong assumption | read from the live mapping, never from config; test against a 0.6.0 fixture index of each engine |
| Two spaces double disk during a swap | full disk at 1M | `bytes_estimate` shown before the start (dim x 4 B x 1.1 Lucene factor x docs); the start refuses when the OpenSearch data path lacks 2x the estimate (fail closed) |
| Query p95 doubles under the build's writes (measured 2x at 200 inserts/s) | slow search during a swap | the job's write rate is a parameter (default 200 docs/s); EP-7 measures it |
| A jittered or synthetic 1M result read as a real one | wrong index decision | EP-3 uses only D6, D7, D8 and set P; the #414 jittered rows stay labelled |
| The #347 candidate's bf16 requirement | no TensorRT FP16 engine | stage D's bf16 or onnxruntime path; refuse the fp16 baker |
| Vendor numbers unreproduced | wasted windows | gates A4 and A6 before any scenario |
| Licence drift of optional models | a shipped optional model becomes non-compliant | the catalog row records the licence and revision read; ATTRIBUTION updated with it |
| Valkey cache poisoning across projects | a query vector from one space used in another | the key includes `space_key`; vectors carry the space key and the builder refuses a mismatch (A8) |
| OpenSearch 3.9 bump | mapping or plugin behaviour change | only after EP-3, as its own pinned bump with the vector-store parity gate (kNN overlap@10 >= 0.95 against the 3.6 index) |

## 10. Owner questions

| ID | Question | Recommendation |
|---|---|---|
| J1 | Put the registry, the reader conversion and the re-index job (lanes ES-A, ES-B, ES-C) into v0.7.0? | yes, all three: the rebuild is the cheapest moment, and #365 then needs no rewrite. Fallback: ES-A and ES-C in v0.7.0, ES-B in v0.8.0 |
| J2 | Keep the 0.6.0 physical names (`pe_embedding`, `embedding_state`, `region_box_embeddings`) as registry data, or rename them in the upgrade? | keep: a rename rewrites every vector of every project in the upgrade for no behaviour change |
| J3 | Delete the in-process `torch` and `onnx` text backends (D-69 makes Triton required)? | yes, in W4 lane 4B |
| J4 | Bump OpenSearch 3.6 to 3.9 (`half_float`, 2- and 4-bit SQ) once EP-3 has numbers? | after EP-3, as its own change with the parity gate; not inside v0.7.0 |
| J5 | Does this plan own the encoder and frame-search experiments (RQ-3b/E-4, RQ-10/E-12), with the clustering plan consuming the verdicts? | yes (section 11); one owner per experiment, vectors shared |
| J6 | The real 1M set: all of COCO train2017 plus val2017 and Open Images validation crops (public; about a 20 GB download and a few GPU hours), plus set P (private)? | yes |
| J7 | Combine across different spaces: copy only equal space keys and defer the rest, or refuse the combine? | copy and defer (no data loss, explicit), with the preview naming the affected counts; refuse only when the target has no space for a target the source has |
| J8 | May the PE text tower move to a GPU TensorRT plan if EP-6 passes, with the CPU plan kept as the selectable option? | yes, per deployment setting |
| J9 | VRAM budget for an optional embedder (2 GiB text, 4 GiB vision), and PE-Core-G as an optional quality space? | budget yes; PE-Core-G not offered unless asked |
| J10 | Core `visual_search_*` indexes stay outside the registry in v0.7.0? | yes; revisit when the core API gets projects |
| J11 | Is the in-product evaluation a hard gate for a flip (with `force=true` to waive), or advisory? | hard gate with the waiver |
| J12 | New box spaces as separate nested siblings (write isolation, more nested docs) rather than a second leaf? | separate siblings |
| J13 | Adopt the #347 plan's calls as a block: staged A-D with the stop rule; TensorRT only for a component that passed; reference arms 2B and 1B on the screen, S2a and S5; GPUs 0 and 2 under one lock in the research windows; generic names in tracked files with this file and the eval note allowlisted; shipping as an optional space through section 5 (not under #365); publish CUB/SOP aggregates as WB2 did; the catalog's default VLM for captions; the eval plan's bars plus gates A5 and A6; multilingual S7 skipped | yes |
| J14 | Close #435 if EP-2's bar fails, or build a frame route over the images index if it passes? | as stated; no third `target` value |

## 11. Required changes to other plan files (for the integration agent)

| File | Change |
|---|---|
| [v070_rebuild/02a_postgres_schema.md.md](02a_postgres_schema.md) | new section "Embedding registry" with the four tables of 5.2 and their RLS; `project_settings` gains `search jsonb NULL`; the creation-order note lists them after `jobs`; the immutability trigger on `embedding_model_versions` |
| [v070_rebuild/02_control_plane_database.md.md](02_control_plane_database.md) | data-ownership table: PostgreSQL owns encoder identity, spaces and their status; OpenSearch keeps the vectors and per-item state fields; section 4.7 lists the registry-named vector fields |
| [v070_rebuild/02b_jobs_locking_events.md.md](02b_jobs_locking_events.md) | kind table row `embedding_reindex` (queue `jobs`, exclusive group per space, auto-resume, 24 h) |
| [v070_rebuild/03a_stage1_foundation.md.md](03a_stage1_foundation.md) | lane 1A creates the registry tables in `0001_control_plane` |
| [v070_rebuild/03b_stage2_control_data.md.md](03b_stage2_control_data.md) | 2A: three default spaces in the create transaction; 2E: spaces per upgraded project with `knn_method` from the live mapping, and a test for each engine |
| [v070_rebuild/03d_stage4_job_families.md.md](03d_stage4_job_families.md) | new lane ES-B (`b2/embedding-reindex`), merge after 4A (it reuses the reprocess decode path) |
| [v070_rebuild/03f_stage6_scheduler_scaleout.md.md](03f_stage6_scheduler_scaleout.md) | new lane ES-C (`b2/embedding-readers`) in W2.4 beside 6A and 6B; 6D documents spaces |
| [v070_rebuild/05_b4_stateless_api.md.md](05_b4_stateless_api.md) | lane 4B adds ES-E (in-process text backends deleted, two env vars retired); 4A-5's `_msearch` takes the space from the binding |
| [v070_rebuild/00_master_plan.md.md](00_master_plan.md) | the lane rows above in W2.1-W2.4 and W4; section 10 names this plan beside the GPU and clustering research plans |
| [v070_rebuild/reference/decision_register.md.md](reference/decision_register.md) | register J1-J14 as decisions once answered; section 9 docs list gains the registry docs |
| [v070_rebuild/reference/traceability.md.md](reference/traceability.md) | #347: research after the tag under this plan (EP-4); #365: seams in v0.7.0 (ES-A..ES-C), flip in v0.8.0 F9; #435: decided by EP-2; #436: v0.8.0 F2 with `space`; the deleted issue-347 plan mapped here |
| [v070_rebuild/04_b3_ultralytics_one_version.md.md](04_b3_ultralytics_one_version.md) | GPU gate G7 (embeddings) records the PE engine parity into the catalog (`engine_parity_min_cos`); below 0.999 the catalog gets a new version instead of an in-place engine swap |
| [v070_rebuild/reference/environment_and_setup.md.md](reference/environment_and_setup.md) | retire `OP_PE_TEXT_BACKEND` and `OP_PE_TEXT_ONNX_PATH` (ES-E) |
| [09b_research_clustering.md](09b_research_clustering.md) | RQ-3b/E-4 and RQ-10/E-12 run here (EP-4, EP-2) and the clustering plan reads their verdicts; every cluster run and centroid store records the `space_key` it clustered; correct its retrieval pin from `coco_bench_2000.json` to `docs/benchmarks/retrieval_baseline/sample_ids.json` |
| [10_v080_platform_breadth.md](10_v080_platform_breadth.md) | F9: the setting, mapping, migrate job and routes now exist from v0.7.0 (ES-A..ES-C); F9 becomes "register the winning model version, build, evaluate, flip", re-based on Celery and PostgreSQL (no lease), size M instead of XL; F2: `group_by` composes with `space`; Q-8 answered by J7 |
| [README.md](../README.md) | research table row for this plan (done in this commit); the issue-347 row removed (done) |
| [embedding_model_eval_plan.md](../../research/embedding_model_eval_plan.md) | its section 3.1 harness location is superseded by 6.1 here; results still go to its section 10 |
| `scripts/codegen/naming_leak_allowlist.txt` | the entry for this file (done in this commit); the entry for the deleted issue-347 plan removed (done) |
