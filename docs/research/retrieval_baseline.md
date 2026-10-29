# Retrieval quality baseline

Measured 2026-10-29 on a test stack (compose project `optest1`, API on loopback port 5603, version
0.6.0 images). First measured baseline of search quality; no search code was changed. Public data
only: COCO val2017 images and the human COCO captions and instance annotations (captions CC BY 4.0,
images under Flickr terms).

Harness: `scripts/bench/retrieval_eval.py` (+ `_data`, `_metrics`, `_runs`). The metric code is
covered by `tests/test_retrieval_eval_metrics.py` (hand-computed values; a wrong nDCG formula
fails it). Raw output of every run: `docs/benchmarks/retrieval_baseline/*.json`.

## Setup

- Sample: 2,000 val2017 images drawn with seed 20261028 from images having at least 5 captions
  (ids, file names and sha256 per image are in `docs/benchmarks/retrieval_baseline/sample_ids.json`; the captions are not copied into the repo, `prep` re-derives them from the official annotations).
  Text queries are the human captions: 2,000 queries (first caption of each image) and 10,000
  (all 5).
- Two search paths were measured on the same 2,000 images:
  - core `POST /search/text` and `POST /search/image` (MobileCLIP, global index; one vector per image);
  - curation `GET /curation/projects/{project}/search/text` (PE-Core-L14-336 text tower against
    per-crop image embeddings of a throw-away project ingested through `ingest/upload`).
- Text-to-image truth: the image the caption belongs to (exactly one relevant image per query, so
  AP equals reciprocal rank and mAP@10 equals MRR@10 below). 95% intervals are bootstrap over
  queries (1,000 resamples).
- Curation results are crops, so an image is ranked at the first appearance of any of its crops
  (pages of 200, up to 5 pages). The crop-level rank is also stored in the raw JSON.
- Image-to-image truth is a proxy: another image is relevant when the COCO category sets have
  Jaccard at least 0.5. It is not a human relevance judgement. Leave-one-out; the query image is
  removed from its own result list.
- Load: the host was shared and busy (1-minute load average 13 to 102 during the runs, recorded as
  `host_load_start/end` and `load1_samples` in each JSON), and all runs used `nice -n 10`. Latencies
  are therefore upper-bound-ish and noisy; quality numbers are deterministic (identical across reps).

## Text to image (one relevant image per query)

| Path | Queries | recall@1 | recall@5 | recall@10 | MRR@10 (= mAP@10) | nDCG@10 | not in top 100 |
|---|---|---|---|---|---|---|---|
| core, MobileCLIP | 2,000 | 0.567 [0.546, 0.588] | 0.817 | 0.886 [0.872, 0.901] | 0.671 | 0.723 | 11 |
| core, MobileCLIP | 10,000 | 0.591 [0.581, 0.600] | 0.824 | 0.896 [0.890, 0.902] | 0.691 | 0.740 | 37 |
| curation, PE-Core crops | 2,000 | 0.383 [0.361, 0.403] | 0.628 | 0.733 [0.715, 0.752] | 0.488 | 0.547 | 67 |
| curation, PE-Core crops | 10,000 | 0.401 [0.392, 0.411] | 0.642 | 0.738 [0.729, 0.746] | 0.504 | 0.560 | 333 |

Sources: `text-core_c1_r1.json`, `text-core_c5_r1.json`, `text-curation_c1_r1.json`,
`text-curation_c5_r1.json` (field `quality`).

Reading the numbers:

- The core path finds the captioned image at rank 1 for 57-59% of captions and in the top 10 for
  about 89%. Failures are mostly near-synonymous scenes (a different baseball player image outranks
  the right one); see `failure_examples` in the JSON.
- The curation path scores lower on this task (recall@10 0.73 vs 0.89), but the two are not an
  equal comparison: curation ranks crops (the default pipeline produces many boxes per image, mean
  113.7 distinct images in the first 200 hits), so one image competes with the crops of 1,999
  others and its whole-scene matches are diluted. Whether PE-Core or MobileCLIP is the better
  encoder cannot be concluded from this table; the experiment that separates them is whole-image PE
  embeddings on the same sample (not done).
- Five captions per image give slightly higher numbers than the first caption alone (the first
  caption is not special; the extra ones are different phrasings), with tighter intervals.
- `min_score=0.2` on core text search changed nothing (`text-core_c1_minscore0.2.json` has
  identical quality to `c1_r1`; no query returned empty).

## Image to image (category-overlap proxy)

From `image-core_r1.json` (1,982 query images that have at least one COCO category):

| Metric | Value |
|---|---|
| precision@10 (relevant = Jaccard >= 0.5) | 0.476 |
| same, normalised by the best achievable (relevant count in pool capped at 10) | 0.507 |
| chance precision@10 (random ranking) | 0.015 |
| queries with zero relevant in top 10 | 217 of 1,982 |
| queries with no relevant image anywhere in the pool | 21 |
| query itself returned at rank 1 | 1,982 of 1,982 |

The proxy under-counts true visual similarity (two images of similar scenes with different annotated
objects count as misses); treat it as a relative yardstick for comparing encoders or settings, not
an absolute quality figure. The failure examples in the JSON show cluttered indoor scenes whose
category sets are large, where Jaccard 0.5 is hard to meet.

## Latency (3 reps each, shared busy host)

| Route | Reps p50 / p95 ms (rep1, rep2, rep3) | Source |
|---|---|---|
| core `POST /search/text` (2,000 queries/rep) | 41.4/54.1, 41.4/58.2, 39.3/53.5 | `text-core_c1_r{1,2,3}.json` |
| core `POST /search/image` (incl. upload and embed, 1,982/rep) | 60.9/71.7, 64.5/80.6, 61.7/76.1 | `image-core_r{1,2,3}.json` |
| curation `GET search/text` (first page of 200) | 119.6/172.6, 114.3/162.6, 116.5/163.2 | `text-curation_c1_r{1,2,3}.json` |

Precision/recall were identical across reps, so only timings vary. Host load during the reps is
in each JSON; rep 1 of the image runs began at load 102, so its latency is the least trustworthy.

## Curation filter and pagination behaviour

From `filters_r1.json` (ten classes, 30 queries each; ten clusters; 20 pagination checks):

- Correctness: 0 items violating the `class_name` filter in 300 queries, 0 violating `cluster_id`,
  0 score-order violations, 0 empty results (every filtered query returned 10 items).
- Filtered top-10 agrees with the top-10 of the unfiltered 200-deep list restricted to that class on
  0.96-1.0 mean overlap (only queries where the deep list held at least 10 matches are comparable;
  the count per class is in the JSON, as low as 2 for `cup`).
- Pagination: pages of 20 concatenate to exactly one page of 60, with no repeated items and no score
  rising across pages (20 of 20 queries). `min_score=0.3` returned no item below 0.3.
- Validation: `page_size` 201, empty `q`, `page=0` and `min_score=1.5` return 422; an unknown
  `class_name`, an unknown `cluster_id` and a page past the retrieval depth return 200 with no items.
- Latency finding: filtered searches took p50 of roughly 80-94 ms for eight classes versus about
  18 ms unfiltered, while `bottle` and `bowl` filtered took about 23 ms and cluster-filtered
  searches about 17-21 ms. The cause was not investigated (a single rep under load); it is worth a
  separate look at how the class filter is applied in the kNN query.

## Near-duplicate grouping at ingest

From `dedupe_r1.json` and `core_ingest.json`: 300 originals were turned into variants (JPEG q60,
half-size, 90% centre crop, brightness x1.15), 100 per threshold, and ingested through
`POST /ingest` with near-duplicate detection.

| Threshold | q60 | half-size | 90% crop | brighter |
|---|---|---|---|---|
| 0.99: grouped (with original) | 34 (24) | 13 (9) | 36 (21) | 96 (87) |
| 0.95: grouped (with original) | 100 (55) | 100 (61) | 99 (75) | 100 (87) |
| 0.90: grouped (with original) | 100 (52) | 100 (52) | 100 (67) | 100 (71) |

Each cell is out of 100. "Grouped with original" is lower than "grouped" because variants also join
a group of their own earlier variants without the original appearing as a member, and (at 0.95 and
0.90) because distinct COCO images can themselves exceed those thresholds. Findings: at the default
0.99 a plain recompression or resize is mostly *not* caught (34 and 13 of 100), while 0.95 catches
nearly all transforms but the grouped set then mixes in genuinely different images; the 2,000
distinct originals produced one false group at 0.99 (similarity 0.9924). Exact re-upload of 20
originals returned `duplicate` for all 20.

## Limits

- One sample of 2,000 images and one seed; category-overlap truth is a proxy.
- The index held 2,000 images (curation: 2,000 images, the crops of the default region cascade),
  small relative to the 1M-image target; recall at that scale is not inferred from these numbers.
- The curation and core encoders differ, and crops versus whole images differ, so the two text rows
  are a product-path comparison, not an encoder comparison.
- The host was loaded throughout; timings are indicative.

## Reproduce

```bash
.venv/bin/python scripts/bench/retrieval_eval.py prep --work <scratch> --n 2000
.venv/bin/python scripts/bench/retrieval_eval.py ingest-curation --work <scratch> --project <throwaway>
.venv/bin/python scripts/bench/retrieval_eval.py ingest-core --work <scratch>
.venv/bin/python scripts/bench/retrieval_eval.py text-core --work <scratch> --captions 1 --tag c1_r1
# also: text-curation, image-core, filters, dedupe; core-cleanup removes the sample from the core index
```

The throw-away project `rq-ret1` and the sample's core-index entries were deleted after the runs.
