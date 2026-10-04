# Market research and competitive analysis

**Summary.** OpenProcessor (backend) and Cropwright (web UI) are a self-hosted, any-domain loop:
ingest, detect, cluster, label with a vision-language model (VLM), confirm by a person, train,
compare, promote back to serving. This folder compares that loop against open-source annotation
tools, commercial labeling platforms, foundation-model auto-labelers, curation tools and
training/MLOps products. Written 2026-10-04 for OpenProcessor v0.4.0 and Cropwright 0.1.0.

## Contents

| File | What it covers |
|---|---|
| [01-landscape-overview.md](01-landscape-overview.md) | Market map, trends, demand drivers, pricing models, market-size figures |
| [02-open-source-tools.md](02-open-source-tools.md) | Label Studio, CVAT, Labelme, X-AnyLabeling, FiftyOne, LightlyStudio and others |
| [03-commercial-platforms.md](03-commercial-platforms.md) | Roboflow, Labelbox, Encord, SuperAnnotate, cloud labeling services and more |
| [04-auto-labeling-and-foundation-models.md](04-auto-labeling-and-foundation-models.md) | Autodistill, Grounded-SAM, SAM 3, VLM labeling |
| [05-curation-and-active-learning.md](05-curation-and-active-learning.md) | FiftyOne, Lightly, Cleanlab, Encord Active, Aquarium, Galileo |
| [06-feature-comparison-matrix.md](06-feature-comparison-matrix.md) | One big table, all products |
| [07-ease-of-use-and-workflow-comparison.md](07-ease-of-use-and-workflow-comparison.md) | Time to first dataset, install effort, review UX, step-by-step workflows |
| [08-training-import-export-comparison.md](08-training-import-export-comparison.md) | Training, import and export formats per tool |
| [09-gap-analysis-and-positioning.md](09-gap-analysis-and-positioning.md) | Differentiation, gaps, risks, roadmap, messaging, target users |
| [10-sources.md](10-sources.md) | Every URL with access date, plus what is unverified |

## Executive summary

- No single product found combines all of: self-hosted, open-source, name-based class identity,
  per-project isolation, detector then embedding clusters then VLM suggestion then human
  confirm, and train, bake-off and promote to a served model. Individual pieces exist
  everywhere; the closed loop in one self-hosted stack is rare.
- The nearest neighbours each cover about half: CVAT (annotation breadth, video, SAM/YOLO
  assist, RBAC; no clustering or train/promote), FiftyOne (curation math, embeddings; weak
  annotation, polished multi-user in paid tiers), LightlyStudio (embeddings, sampling, SAM
  plugins; no VLM labeling or training ops), Roboflow and Ultralytics Platform (annotate,
  auto-label, train, deploy; hosted and credit-priced), X-AnyLabeling (desktop, many models,
  GPL-3.0).
- OpenProcessor trails clearly on annotation tooling breadth (boxes only as first-class
  geometry; no polygon editing, keypoints, video, 3D), collaboration (Cropwright has no login),
  polish, ecosystem and format coverage (YOLO, COCO and own export only).
- Licensing is a real constraint: AGPL-3.0-or-later, and it vendors an AGPL Ultralytics fork.

## Key takeaways

1. **Real gap, narrow wedge.** The wedge is "bulk-label by cluster and VLM, confirm at keyboard
   speed, then train and promote, self-hosted, any domain". It is not "better annotation tool".
2. **Competitors do better** at geometry types, video, multi-user workflow, polish, hosted
   convenience and ecosystem size.
3. **Trend favours the idea.** Auto-labeling with SAM 3 and VLMs is now table stakes in
   Roboflow, CVAT, X-AnyLabeling and Ultralytics Platform; the human-confirm loop is where
   differentiation must come from.
4. **Commercial landscape is unstable** (acquisitions, shutdowns, pivots to LLM data), which
   helps a self-hosted option on trust and continuity.
5. **Biggest risks:** AGPL adoption friction, heavy hardware and install footprint (about
   60 GB for `core`, 16 GB VRAM), no auth, and unpublished benchmarks.

## Verdict: where we win / where we trail / what is unique

| | Assessment |
|---|---|
| **Win** | Self-hosted end-to-end loop; project isolation at the storage layer; class identity by name across import, combine, export, train, promote; the lock rule (automation never overwrites human labels); editable prompt packs and region profiles as versioned data; open-vocabulary SAM 3 stage; selectable local or remote VLM with consent gate; one-line installer with digest-pinned images |
| **Trail** | Annotation geometry and editing; video; multi-user, roles, SSO, audit; format breadth; maturity and community; hosted option; published benchmarks; documentation of accuracy gains |
| **Unique (as far as found)** | Detector, clusters, VLM, human confirm, train, bake-off, promote in one self-hosted open-source stack with name-based class identity and per-project isolation. "As far as found" is a statement about this survey, not a proof |

## Methodology and limits

- Product facts about OpenProcessor come from this repository (README, INSTALLATION.md,
  docs/CURATION.md, docs/PERFORMANCE.md, docs/VISION_AND_GOALS.md, LICENSE) at origin/main
  for v0.4.0, plus Cropwright's public description. Evidence is cited by file in
  [10-sources.md](10-sources.md).
- Competitor facts come from vendor and GitHub pages and web search summaries fetched on
  2026-10-04. Page text was summarized by a tool, not read in a browser; star counts and
  prices move daily. Treat numbers as snapshots.
- "Unverified" marks claims seen in one secondary source, vendor marketing, or a page that
  would not load (for example some vendor pricing pages returned errors or truncated content).
- Nothing was benchmarked. No head-to-head accuracy or speed comparison was run. Vendor
  speed-up claims are not repeated as facts.
- Market-size figures differ several-fold between analysts; see file 01.
- Scope is image object detection and dataset curation. Text-only, audio and LLM-data
  tools are mentioned only where they shape the market.
- No private datasets or private project content were used.

## Where this folder could be linked from

Suggested, not applied: a line in `docs/README.md` and the "Why" section of the root README,
and a docs-site "About" page.

Last updated 2026-10-04. Sources: see [10-sources.md](10-sources.md).
