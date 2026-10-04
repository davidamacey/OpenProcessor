# 07 - Ease of use and workflow comparison

**Summary.** Hosted tools win time to first labeled image (minutes, no install). Pip-installable
curation tools are next. OpenProcessor has the heaviest install (GPU, Docker, tens of GB) but
its bulk-label loop can be faster per item once images are ingested. Time estimates below are
reasoned from documented install steps, not measured, except where the repository reports a
measurement.

## Time to first labeled dataset (estimates, unmeasured except where cited)

| Product | Install | First labels | Basis |
|---|---|---|---|
| Roboflow, Ultralytics Platform, Labelbox, Encord | Sign up | Minutes | Hosted; Roboflow Auto Label from text prompts |
| CVAT | `git clone` and `docker compose up -d` | Tens of minutes | Documented 2-command start; labeling is manual or model-assisted per image |
| Label Studio | Docker or pip | Tens of minutes | Config XML then import |
| Labelme / X-AnyLabeling / Make Sense | pip, binary or browser | Minutes | Single-user |
| LightlyStudio / FiftyOne | `pip install` | Minutes to index a folder | `lightly-studio quickstart` |
| **OpenProcessor + Cropwright** | One-line installer: 30-60 minutes first install (image pulls and TensorRT export, per README); about 60 GB disk for `core`, about 135 GB with every tier | Ingest speed measured once: 13.39 images/s on a 2,000-image COCO subset on one 48 GB GPU, 4.90 images/s on a mixed set with half 12-20 MP JPEGs (docs/PERFORMANCE.md; single run, narrow vehicle detector) | Plus VLM labeling time, which is not measured in this repository's docs |

## Workflow, step by step

| Step | Annotation tools (CVAT, Label Studio) | Hosted platforms (Roboflow, Ultralytics) | Curation tools (FiftyOne, Lightly) | OpenProcessor + Cropwright |
|---|---|---|---|---|
| Import | Upload or cloud storage; many formats | Upload; many formats | Index folder or COCO/YOLO | Ingest images; import YOLO, COCO, own export with preview, name mapping, undo |
| Label | Draw or click-assist per image | Manual plus Auto Label (credits) | Mostly external labeler | Detector then optional SAM 3 and VLM propose; person confirms by cluster or box, with hotkeys |
| Review | Task review (some paid) | Review queue | Explore embeddings | Review queues, lock rule, undo |
| Train | External | Built in, hosted GPU | External | Built-in YOLO26 trainer, campaign, bake-off, frozen holdout |
| Export | Many formats | Many formats | Export | YOLO exports with manifest SHA |
| Deploy | External | Hosted or self-hosted Inference | External | Promote to Triton serving, gated |

## Review UX

- CVAT, Labelme, X-AnyLabeling: strongest for geometry editing per image.
- OpenProcessor: strongest for "confirm many near-identical items at once" (cluster grid, bulk
  actions, configurable keymaps); not designed for drawing precise polygons.
- Cropwright is documented as having no login; multi-annotator coordination is out of scope in
  v0.4.0.

## Honest assessment against the leaders

- Setup: far heavier than anything hosted or pip-based; GPU and Docker are hard requirements.
- Breadth: far narrower than CVAT or Label Studio on geometry and data types.
- Loop: more integrated than any open-source competitor found, and comparable to the
  hosted Roboflow and Ultralytics flows, but self-hosted.
- Evidence: the claim "faster to a trained detector" is plausible but unmeasured; publishing a
  public-data time-to-model benchmark is the missing proof (file 09).

Last updated 2026-10-04. Sources: [10-sources.md](10-sources.md); OpenProcessor README.md and docs/PERFORMANCE.md.
