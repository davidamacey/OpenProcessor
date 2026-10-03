# Storage sizing and ingest baselines

Status: **partially done**. Issues: #45 (public COCO baseline set and results in
`docs/PERFORMANCE.md`), #56 (quantization and retention benchmarks), #53 (shard budget per
project). The measurements below are done and still true for the default schema. Open: publish a
re-measured table from the public baseline set in `docs/PERFORMANCE.md`, benchmark quantization
on real embeddings, and add retention and index-partitioning guidance.

Measured on OpenSearch 3.6 with the k-NN plugin (faiss HNSW, cosine), a 1024-dimension
float32 embedding per vector, derived source on (the OpenSearch 3.x default), no replicas.

## 1. Headline numbers

- One 1024-d float32 vector costs **8.4 KB** on disk (2.0x its raw 4 KB): about 4.2 KB for the
  HNSW search structure plus 4.1 KB for the flat copy used for read-back, merges and exact
  re-scoring. Measured at 3,000 docs (8,357 bytes per doc) and 10,000 docs (8,359 bytes per doc)
  after merge.
- An item (one detection with one vector and its metadata) is about **10 KB**; an image document
  with its whole-image vector is about **8.7 KB**. Non-vector metadata is about 850 B per item and
  350 B per image. Vectors are about 93 percent of the bytes.
- Each embedded region box adds about 1 KB of metadata plus 8.4 KB for its vector.
- Replicas multiply storage by (1 + replicas). Single-node deployments use no replicas.

## 2. Why a vector costs more than 4 KB, and the layout rule

Three copies are possible: the search structure, the flat copy, and, only when derived source is
off, a JSON text copy inside `_source` (about 17 KB). Keep derived source **on** so only the first
two exist (8.4 KB). With it off the same index measured about 25 KB per vector.

One query form misbehaves with derived source on: a search or scroll whose `_source` list names
the bare nested field (`region_box_embeddings`) returns the number `1` instead of each vector.
Leaf paths (`region_box_embeddings.embedding`), GET, mget, inner_hits, partial `_update` and
`_reindex` all return real vectors. The code uses one helper, `box_vector_source_includes`, that
returns leaf paths. Do not drop vectors with `_source.excludes`: a partial `_update` then loses
them. Indexes created before the derived-source fix keep the old static setting until recreated.

Measurement lesson: right after a load or force-merge, segment files held by open readers inflate
the size (one measurement showed 4x). Merge twice, then refresh, flush and wait about 10 s before
measuring.

## 3. Measured data

Fresh default-schema project, 4,000 real photos (a small-photo half and a large-JPEG half,
12 to 20 MP), default pipeline (whole-image vector, vehicle detections, one crop vector per
detection, region stage with a vector per accepted box), settled after two forced merges:

| Index | Docs | Vectors | Bytes | Bytes per doc | Bytes per vector |
|---|---|---|---|---|---|
| images | 4,000 | 4,000 | 34,770,580 | 8,693 | 8,344 |
| items | 5,345 (7,062 Lucene docs incl. nested) | 6,091 (5,345 crop + 746 region box) | 55,383,310 | 10,361 | 8,347 |
| classes, configs, labels, umap_state | 2 | 0 | 12,683 | n/a | n/a |
| Total | | 10,091 | 90,166,573 | | |

Per image: 22.5 KB total (images 8.7 KB plus 1.336 items at 10.4 KB). Inside the items shard the
file sizes were .faiss 25.9 MB, .vec 25.0 MB, stored fields 2.4 MB, doc values 1.1 MB, terms
0.5 MB, points 0.4 MB.

A 2,000-image public COCO subset (default pipeline, 2,741 crops) settled at 44.6 MB total: items
27.2 MB, images 17.4 MB, everything else under 13 KB.

A production-scale reference deployment (259,420 images, 560,099 items, 2.16 items per image, 62
percent of items carrying two vectors) held 10.4 GB of indexes: items 8.34 GB (14.9 KB average),
images 2.15 GB (8.3 KB). The per-vector formula reproduces it, so size is linear in detections per
image.

## 4. Estimating your deployment

Per image (efficient layout, no replicas, no quantization): `8.7 KB + K x 10.4 KB`, where K is the
average detections per image. Add `8.4 KB + 1 KB` for every embedded region box.

| Detections per image (K) | Per image | 1M images/day | 30 days | 1 year |
|---|---|---|---|---|
| 1.34 (measured) | 22.5 KB | 22.5 GB | 0.68 TB | 8.2 TB |
| 5 | 60.5 KB | 60.5 GB | 1.8 TB | 22 TB |
| 10 | 112 KB | 112 GB | 3.4 TB | 41 TB |

The source images dominate: this sample averaged 3.4 MB per file (6.6 MB for the large half),
150 to 300 times the database bytes per image. Plan the image store separately.

Search memory: the HNSW graph stays in native RAM when a vector index is searched, about
`1.1 x (4 x 1024 + 8 x 16)` = 4.6 KB per vector (the k-NN plugin's formula; verify against the
current OpenSearch documentation). The default pipeline produced about 2.5 vectors per image in
the production-scale reference, so graph RAM, not disk, is the practical limit long before
storage.

## 5. Recommendations (to document and, where noted, benchmark)

1. Keep derived source on and never store vector text in `_source`.
2. Store each vector once per document.
3. Partition by time (daily or weekly indexes per project) with a retention policy; cap each
   shard at a few tens of GB. (Open: design and shard cost per project is #53.)
4. Embed only what you need: skip whole-image vectors when only detections are searched; skip
   tiny or low-confidence detections; skip region-box embeddings unless clustering them. (See
   `docs/design/generic_detector_and_selective_embedding_plan.md`.)
5. Quantize only after a benchmark on real embeddings. Tests on random vectors (fp16 about
   6.3 KB per doc; `on_disk` about 3.7 KB per doc at recall 0.75; 4x compression at recall 0.18)
   are not representative. (Open: #56.)
6. No replicas on a single node.
7. Keep documents lean: cap history fields, keep thumbnails and text blobs out of the index.
8. Give OpenSearch an adequate heap (a heap-dump file from a crash once took 2 GB of a data
   volume).

## 6. Ingest and pipeline baseline (to repeat for #45)

One run, not averaged. Stack: one 48 GB GPU hosting the detection models, API and segmenter;
a second 48 GB GPU hosting the VLM; OpenSearch heap 8 GB; API with 32 workers. Region profile: a
small-region example profile with no parent-class scoping (so every item went to the segmenter,
which is why a parent-scoped profile is the right default, see #46). Item classes: car,
motorcycle, bus, truck from the COCO detector.

| Stage | 4,000-image mixed sample | 2,000-image public COCO subset |
|---|---|---|
| Ingest (`POST /ingest/batch`, 32 images per request, 4 client threads) | 4.90 images/s (small-photo half about 15, large-JPEG half about 2.5; decode and resize of 12 to 20 MP is the cost); 0 failed | 13.39 images/s, 0 failed; 2,741 crops (1.37 per image); batch p50 10.0 s |
| Region stage (segmenter proposal plus VLM verify), overlapped with ingest | 3.1 crops/s overall, 3.7 crops/s after ingest finished (VLM bound); 5,345 crops in 1,709 s | about 4.1 crops/s; segmenter 25 percent hit rate, 2.5 to 3.7 s per call |
| Region outcomes | 746 detected, 225 verify_rejected, 1,403 no_region_visible, 2,971 no_region_box | 214 detected, 162 verify_rejected, 1,218 no_region_visible, 1,147 no_region_box |
| VLM class labeling (region stage off) | n/a | all 2,741 items in about 7 min |
| Clustering: auto-label with `recluster_unvalidated=true` (IVF, CPU, 1024-d) | 98.7 s over 5,345 items (512 clusters) | 65.8 s over 2,741 residuals (456 candidate clusters) |
| Region clustering | about 6 s over 746 boxes | about 4 s over 214 boxes |

GPU memory at the end of the first run: 25.9 GB on the detection GPU, 21.2 GB on the VLM GPU.
The cluster-refresh worker fires during ingest (every 200 new crops), so its runs are inside the
ingest and worker times. To compare after an optimization (#40), repeat with the same sample
manifest (fixed seed), a fresh project slug each time (a deleted project's slug is retired),
poll the region drain count until zero, then measure storage after two forced merges.

## 7. What is open

- Build the public baseline fetcher and pinned manifests (2k, 10k, 50k, 100k COCO images) and
  record the table in `docs/PERFORMANCE.md` (#45).
- Re-measure storage on that set and publish the table in `docs/PERFORMANCE.md`, the README and
  docs-site (#45).
- Quantization and recall benchmark, k-NN memory formula check (#56).
- Shard cost per project and index sharing policy (#53).
