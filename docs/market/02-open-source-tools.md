# 02 - Open-source tools

**Summary.** Open-source annotators are strong on drawing tools and formats and weak on
curation-by-cluster, VLM suggestion and train-and-promote loops. CVAT is the broadest; Label
Studio is the most flexible across data types; FiftyOne and LightlyStudio lead on curation.
Star counts and versions are 2026-10-04 snapshots from GitHub pages.

| Tool | License | Stars | Annotation types | Auto-assist | Notes |
|---|---|---|---|---|---|
| [Label Studio](https://github.com/HumanSignal/label-studio) | Apache-2.0 | 28.4k | Images, text, audio, video, time series | ML backend SDK (pre-label, active learning) | Docker, pip, Compose with PostgreSQL. Enterprise-only (per earlier reading of vendor docs, unverified today): bulk labeling, review workflows, full RBAC/SAML, plugins |
| [CVAT](https://github.com/cvat-ai/cvat) | MIT (serverless components and FFmpeg may differ) | 16.9k | Box, polygon, mask, keypoint, cuboid, tag; image, video, 3D | SAM, YOLO, Mask R-CNN via serverless | 20+ formats; roles and tasks; Docker Compose. Online free tier; Enterprise self-hosted from $12,000/yr |
| [Labelme](https://github.com/wkentaro/labelme) | GPL-3.0 | 16.2k | Polygon, rectangle, circle, line, point | SAM, EfficientSAM, YOLO-World, SAM 3 | Desktop (Qt6 in v7). Exports VOC, COCO, JSON |
| [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) | GPL-3.0 | 10.6k | Polygons, boxes, rotated boxes, keypoints, masks, OCR tasks | SAM 1-3, Grounding DINO, YOLO-World, Qwen3-VL, Gemini, GPT, YOLO v5-v12, PaddleOCR | Desktop; v4.0.0 2026-08-05; COCO, VOC, YOLO, DOTA, MOT, others |
| [LabelImg](https://github.com/HumanSignal/labelImg) | MIT | 25.1k | Boxes | none | Archived 2024-02-29; points users to Label Studio |
| [Make Sense](https://github.com/SkalskiP/make-sense) | GPL-3.0 | 3.6k | Box, polygon, point, line | In-browser TF.js models | No install; client-side; limited formats for polygons |
| [VoTT](https://github.com/microsoft/VoTT) | MIT | 4.4k | Boxes, polygons | none | Archived 2021-12-07, unmaintained |
| [Diffgram](https://github.com/diffgram/diffgram) | Diffgram License v2 (custom, "commercial open source") | 1.9k | Image, video, 3D, text, audio | Workflow automation | Self-host, Kubernetes; repo active; check license terms before use |
| [Doccano](https://github.com/doccano/doccano) | MIT | 10.8k | Text only | n/a | Out of scope for vision; mentioned for completeness |
| [Datumaro](https://github.com/open-edge-platform/datumaro) | MIT | 692 | Dataset library, not an editor | n/a | Converts CVAT, COCO, VOC, YOLO, KITTI, others; quality checks |
| [supervision](https://github.com/roboflow/supervision) | MIT | 51.1k | Library | n/a | Dataset load, split, merge, convert (YOLO, VOC, COCO) |
| [FiftyOne](https://github.com/voxel51/fiftyone) | Apache-2.0 | 11.1k | Native 2D and 3D labeling, viewer | Model zoo, Brain | Curation and evaluation leader; Teams/Enterprise tiers add collaboration and scale |
| [LightlyStudio](https://github.com/lightly-ai/lightly-studio) | Apache-2.0 | 892 | Image and video annotation | Plugins (for example SAM) | `pip install lightly-studio`; local-first; claims 2M+ images on an M1 laptop (vendor claim) |

Not covered individually: Supervisely Community edition (free tier, 5 GB / 10,000 files; see
file 03), Annotab (no reliable source found; unverified).

## Observations

- None of the open-source annotators has a clustering-first triage or a train-and-promote loop;
  FiftyOne and LightlyStudio have the embeddings but hand off training.
- Several desktop tools (Labelme, X-AnyLabeling) are GPL-3.0; OpenProcessor is AGPL-3.0, so
  license posture is comparable, but AGPL is stricter for network use.
- Several once-popular tools are archived, which is a continuity argument for maintained
  self-hosted stacks.
- X-AnyLabeling is the nearest "many models in one free tool" competitor, but it is a desktop
  single-user annotator without a project database, clustering or training pipeline.

Last updated 2026-10-04. Sources: [10-sources.md](10-sources.md) section A.
