# 05 - Curation and active learning

**Summary.** The curation space is thinner than annotation. FiftyOne is the richest open
toolkit; LightlyStudio is the closest open product to cluster-and-sample with a UI; Cleanlab
finds label errors. Encord Active (open source) was archived and Aquarium's CV product was
retired, so commercial curation lives mostly inside platforms (Encord Index, SuperAnnotate).

| Tool | Status | License | What it offers | Gap relative to OpenProcessor |
|---|---|---|---|---|
| [FiftyOne](https://github.com/voxel51/fiftyone) | Active, 11.1k stars | Apache-2.0 core | Embedding exploration, similarity search, Brain (mistakenness, edge cases), evaluation, plugins, 2D/3D labeling | Hands off training; collaboration, on-prem and scale in paid Team/Growth tiers (seats from vendor pricing page) |
| [LightlyStudio](https://github.com/lightly-ai/lightly-studio) | Active, 892 stars | Apache-2.0 | Automatic embeddings, similarity, six sampling strategies, object-level search and clustering, annotation, evaluation, roles (Viewer, Labeler, Editor, Admin), COCO/YOLO/VOC export | No train-and-promote; no VLM label suggestion found on its page |
| [Cleanlab](https://github.com/cleanlab/cleanlab) | Active, 11.7k stars | Apache-2.0 | Label-error detection incl. object detection (ObjectLab) and segmentation | Library; no UI loop |
| Encord Active | Archived 2025-08-07 (v0.1.84) | Apache-2.0 | Data and model quality metrics | Use is by fork only |
| Encord Index / Active (cloud) | Commercial | n/a | Curation inside Encord platform | Quote-only, cloud/VPC |
| Aquarium | Team joined Notion; CV curation product (Illume) shut down per search summary | n/a | Was embedding-based curation | Not available |
| Galileo | Now LLM and agent evaluation per search summary | n/a | Dataset curation inside LLM evals | Not a CV curation option today (unverified whether any CV product remains) |
| SuperAnnotate | Commercial | n/a | Data curation, AI/human routing | Quote-only |

## What OpenProcessor ships here (evidence: docs/CURATION.md, docs/research/curation_scores.md)

- Embedding clustering (per item and per region box), refine-by-cluster, review queues, item
  scores, diverse selection and semantic search, all over a shared item filter.
- Selective embedding policy (`all`, `selected`, `lazy`) to control storage and GPU cost.
- Frozen test holdout and the lock rule.

## Honest comparison

- FiftyOne's quality metrics (mistakenness, hardness and similar) and evaluation tooling are
  more mature than OpenProcessor's scores; OpenProcessor's advantage is that curation, labeling
  and training are one product.
- LightlyStudio is the strongest open-source analogue for the curation half; it lacks the
  training half and, on its public page, VLM suggestion.

Last updated 2026-10-04. Sources: [10-sources.md](10-sources.md) section D.
