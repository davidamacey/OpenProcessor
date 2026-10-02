# OpenProcessor: Vision and Goals

This document is the canonical statement of what OpenProcessor is for, what
the v0.4.0 release must deliver, and the standards the codebase is held to.
It exists so that anyone — human or AI agent — working in this repository
can orient quickly: `CLAUDE.md` and `README.md` both link here rather than
restating it.

## The core mission

OpenProcessor is a public, generic backend for computer-vision dataset
curation and training that works for **any image domain** — vehicles,
medical imaging, biology, wildlife, industrial inspection, or anything
else — not hardcoded to one use case. It is not a demo or a proof of
concept: it is meant to be a professional, production-capable application,
released together with its frontend, **Cropwright**, as a matched pair.

The full pipeline, end to end:

```
ingest → primary detection → region stage (SAM3 + VLM, one-or-many
boxes per item) → embedding + clustering → human/AI review and labeling
→ import existing labeled datasets → export → train (YOLO26) → bake-off
→ promote back into production
```

Every stage is designed to be reused across domains by *configuring* it
(detection profiles, prompt packs, class registries) rather than by
forking the code per domain.

**Explicit owner scope decision:** the full v0.4.0 feature set below was
deliberately *not* trimmed to ship faster. Nothing here is a stretch goal.

## What "done" means for v0.4.0

- **Multiple isolated projects.** Each project is its own dataset
  workspace — its own indexes, its own directories, its own class
  registry — with real isolation enforced at the storage layer, not just
  convention.
- **Editable prompt packs.** What the vision-language model is told to
  look for and how to answer is data, not code — editable per project,
  versioned, and revertible.
- **Editable region profiles.** Detection/segmentation configuration
  (which models, thresholds, and heuristics define "a region of interest
  inside this item") is likewise data, not code.
- **One-or-many regions per item.** An item can have zero, one, or many
  detected sub-regions (e.g. a vehicle crop with four wheels, each
  independently found, verified, embedded, clustered, and labeled) — not
  an assumption baked in that every item has exactly one region.
- **Configurable keyboard shortcuts**, per project, for the review UI.
- **Selectable AI labeling model.** Which vision-language model performs
  labeling is a runtime choice (local or remote endpoint), not a
  hardcoded dependency.
- **Import existing labeled datasets** (YOLO, COCO, or this project's own
  prior export format) into a project, with classes mapped **by name,
  never by index** — a dataset's class order is never trusted across a
  boundary.
- **Combine multiple projects** into a new one, preserving source data
  and resolving class-name conflicts explicitly.
- **A one-line installer** that gets a working stack running with no git
  clone, no local image build, and no host Python required.
- **Full documentation** covering every feature above, current as of the
  release — not aspirational.
- **Tested end to end**, including a public, reproducible example (a
  car-photo dataset where the primary detector finds vehicles and the
  region stage finds wheels) that proves the whole pipeline works on data
  anyone can download, not just on private fixtures.

## Why this matters

A fast, correct backend and a genuinely good, domain-agnostic labeling GUI,
released together, is something that does not otherwise exist. The value
of OpenProcessor is that combination — not the backend alone, and not a
generic labeling tool alone. Backend work here is not happening in a
vacuum: every wire contract is frozen and documented specifically so
Cropwright (the frontend) can consume it whenever its own development
session is active, and the whole system is built to be a stable foundation
for the GPU/inference performance optimization work that follows the
release.

## The standards this codebase is held to

- **The backend stands on its own.** It must work correctly as an
  inference engine and API by itself — it does not depend on the frontend
  existing or being complete to be correct, tested, or useful.
- **No skipped or missed bugs, but no wasted scrutiny either.** Every
  change gets the review depth its actual risk warrants: destructive
  operations, concurrency, cross-project isolation, and security-sensitive
  surfaces (secrets, external network calls) get real adversarial review;
  read-only or low-risk surfaces do not need the same weight.
- **No dead code, no duplicate logic, no routes left running in
  parallel with their replacement.** When something is superseded, the
  old version is deleted in the same change, not left "just in case."
  There is no backwards-compatibility layer in this project by design —
  no shims, no migrations, no deprecated-but-still-served endpoints.
- **Class identity is the name, never the raw index**, at every boundary
  a dataset crosses: import, combine, export, train, promote, and
  cross-project model sharing.
- **Fail-closed, not fail-open.** An unbound project, a stale registry, or
  an ambiguous state refuses the operation; it never silently guesses or
  falls back to a default that could leak data across a boundary.
- **Every wave of work is merged incrementally**, with its own gate
  (automated tests, linting, contract-sync checks) and — for anything
  touching data integrity, isolation, or security — an independent review
  before it lands, not batched into one release-day merge.

## What's explicitly out of scope for v0.4.0, and why

These are real, tracked commitments, not abandoned ideas — they are
sequenced deliberately rather than included now:

- **Full-image, open-vocabulary SAM3 detection**
  ([issue #30](https://github.com/davidamacey/OpenProcessor/issues/30)).
  Today, the region stage runs SAM3 *inside* an already-detected item's
  crop, so it can only find sub-regions of something the primary detector
  already flagged. Because SAM3 is open-vocabulary, running it directly on
  the full image would let it discover and label entirely new object
  categories that have no trained detection class yet (e.g. finding "all
  the legos" in a photo with zero prior lego training data). This needs
  its own design pass — data model, storage shape, and a full-image vs.
  crop-region trigger — before implementation, so it's deliberately queued
  after the v0.4.0 release.
- **Deep GPU / Triton inference optimization** (issue #40). Batching
  strategy, GPU utilization, and a zero-copy pipeline (decode once, keep
  data resident on the GPU across models) are real, planned work — but
  they should be tuned against the *final* workload shape. Since
  full-image SAM3 (above) would materially change SAM3's call pattern,
  optimization work is sequenced to follow that design decision, not
  precede it, so it doesn't need to be redone.
- Benchmarks, a technical white paper, and additional domain showcases
  beyond the car→wheel example are explicitly post-release deliverables.

## Where to look for more detail

- `docs/CURATION.md` — the curation subsystem's user-facing feature guide.
- `docs/ARCHITECTURE.md` — component map and system design.
- `docs/design/curation_design_rationale.md` — why the curation subsystem
  is built the way it is.
- `docs/design/curation_api_contract.md` — the full route and wire-model
  reference.
