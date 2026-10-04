# 04 - Auto-labeling and foundation models

**Summary.** Open-vocabulary detectors, SAM-family segmenters and grounded VLMs make first-pass
labels cheap. Most tools expose one or two of them. OpenProcessor combines a project
detector, SAM 3 open-vocabulary and region stages, and a VLM, each editable as data and gated
for cost; what it lacks is a published accuracy comparison.

| Approach | Model / tool | License (verified 2026-10-04 unless noted) | Strength | Limit |
|---|---|---|---|---|
| Distillation pipeline | [Autodistill](https://github.com/autodistill/autodistill) (2.8k stars) | Apache-2.0 | Foundation model labels, target model (YOLO, DETR, ViT) trains; many base models (Grounded SAM 2, GroundingDINO, OWL-ViT, OWLv2, LLaVA, Gemini, others) | Library, no review UI or project store |
| Text-prompted detect + segment | [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) (17.7k) | Apache-2.0 code; checkpoint terms per model (unverified) | Boxes and masks from text | Research-style repo |
| Concept segmentation | [SAM 3](https://github.com/facebookresearch/sam3) | Meta "SAM License" (custom), gated checkpoints | All instances of a noun phrase; text, point, box, exemplar prompts; images and video | Not OSI; weights need access approval |
| Zero-shot detectors | Florence-2 (MIT), OWLv2 (Apache-2.0), Grounding DINO (Apache-2.0 code; 1.5/1.6 Pro paid API) | per search summary; check each checkpoint | Cheap class-name labeling | Accuracy varies; generalists below specialists on COCO zero-shot (secondary source) |
| Grounded VLMs | Qwen3-VL (Apache-2.0, per search summary) | | Boxes by prompt; class reasoning | Box precision and consistency need checking |
| Hosted auto label | Roboflow Auto Label (SAM 3 or Gemini; credit-metered), Ultralytics Platform (SAM 3.1, YOLO smart) | Commercial | One click in the tool | Cloud; vendor says SAM 3 cannot tell fine variants apart (for example crack types) |
| Annotation-tool assist | CVAT (SAM, YOLO via serverless), Label Studio ML backend, X-AnyLabeling, Labelme | See file 02 | Inline click-to-segment | Per-image interactive, not corpus triage |
| Foundation-model tooling for curation | FiftyOne model zoo, LightlyStudio plugins | See file 05 | Run models over a dataset | Not a label-confirm UX |

## Where OpenProcessor sits

Evidence: README feature list, docs/CURATION.md, INSTALLATION.md tiers.

- Primary detector with a full-vocabulary option stored as unlabeled proposals.
- Optional SAM 3 tier: project text prompts run on the whole image; also a crop region stage
  with a hit-rate gate and pause or resume.
- VLM labeling through any OpenAI-compatible endpoint; local vLLM catalog or remote endpoint
  with an explicit acknowledgement that crops leave the host. Verified default is one model;
  other catalog entries are marked unverified.
- Prompt packs and region profiles are versioned data with test-on-crop and impact reports.
- Compared with Autodistill: Autodistill goes straight from labels to a model without a human
  review step; OpenProcessor puts clusters and review between the two, which costs time but
  reduces silent label noise.

## Cautions

- Auto-label quality for fine-grained classes is domain-dependent. No independent benchmark is
  cited here; one vendor page states that SAM 3 cannot distinguish specific variants.
- SAM 3's custom license and gated access affect redistribution and installer UX; the
  OpenProcessor `segmenter` tier requires a Hugging Face token with access.

Last updated 2026-10-04. Sources: [10-sources.md](10-sources.md) section C.
