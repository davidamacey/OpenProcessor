# 06 - Feature comparison matrix

**Summary.** One table per column group, same rows throughout. Legend: Y = documented, P =
partial or paid tier only, N = not found, ? = not verified. "Not found" means not found on the
pages fetched on 2026-10-04, not proof of absence. OpenProcessor cells cite repository docs.

## A. Annotation types and auto-label

| Product | Box | Polygon / mask | Keypoint | Video | Text / OCR | Auto-label (SAM / VLM / detector) |
|---|---|---|---|---|---|---|
| **OpenProcessor + Cropwright** | Y (first-class, one or many per item) | P (SAM 3 mask polygon stored with open-vocab `mask`; no polygon editing documented; imported polygons reduced to boxes) | N | N | P (region text reading via OCR/VLM) | Y: detector, SAM 3, VLM |
| CVAT | Y | Y | Y | Y | tags | Y: SAM, YOLO, Mask R-CNN (serverless) |
| Label Studio | Y | Y | Y | Y (limited) | Y | P: ML backend; auto-label and bulk features partly Enterprise |
| Labelme | Y | Y | Y (points) | N | N | Y: SAM-family, YOLO-World |
| X-AnyLabeling | Y | Y | Y | ? | Y (OCR tasks) | Y: SAM 1-3, Grounding DINO, VLMs |
| FiftyOne | Viewer plus native labeling | Y | ? | Y (viewing) | ? | P: model zoo |
| LightlyStudio | Y | Y | ? | Y | ? | P: plugins (SAM) |
| Roboflow | Y | Y | Y | ? | ? | Y: SAM 3, Gemini, own models |
| Ultralytics Platform | Y | Y | ? | ? | ? | Y: SAM 3.1, YOLO smart |
| Encord | Y | Y | Y | Y | ? | Y (agents) |
| Labelbox | Y | Y | ? | Y | Y | Y (pre-label) |
| SuperAnnotate | Y | Y | ? | Y | Y | Y |
| Supervisely | Y | Y | ? | Y | ? | ? |
| CloudFactory/Hasty | ? | ? | ? | ? | ? | ? |
| SageMaker Ground Truth | Y | Y | ? | Y | Y | Y (automated labeling) |
| Autodistill | via model | via model | N | N | N | Y (is the auto-labeler) |

## B. Curation, review, collaboration

| Product | Clustering / embedding search | Uncertainty / active-learning queues | Human review UX | Multi-user / RBAC |
|---|---|---|---|---|
| **OpenProcessor + Cropwright** | Y (item and box clustering, semantic search) | Y (review sorts, uncertainty, scores) | Y (cluster grid, keymaps, lock rule, undo) | N (no login; LAN-only warning) |
| CVAT | N | N | Y (annotation shortcuts) | Y (roles, tasks; SSO in paid) |
| Label Studio | N | P (active learning via ML backend; webhooks Enterprise) | Y | P (Starter RBAC; SSO Enterprise) |
| Labelme / X-AnyLabeling | N | N | single-user desktop | N |
| FiftyOne | Y | Y (SDK) | viewer-first | P (Team and higher paid) |
| LightlyStudio | Y | Y (sampling) | Y | Y (Viewer/Labeler/Editor/Admin roles listed) |
| Roboflow | ? | ? | Y | P (Teams add-on) |
| Encord | Y (Index) | Y (Active) | Y | Y |
| SuperAnnotate | Y (curation) | Y (routing) | Y | Y |
| Labelbox / Kili / Segments | ? | P (active learning listed for Segments Core) | Y | Y |

## C. Formats, training, serving, hosting, cost

| Product | Import formats | Export formats | Training built in | Model promote / serving | Self-host / offline | License / cost | Install effort |
|---|---|---|---|---|---|---|---|
| **OpenProcessor + Cropwright** | YOLO, COCO, own export (others via conversion) | YOLO (multi-class, single-class) | Y (YOLO26 via trainer; bake-off; holdout) | Y (promote to Triton, gated) | Y (fully; local VLM) | AGPL-3.0-or-later; free; hardware cost | One-line installer; 60-135 GB disk; GPU; 30-60 min |
| CVAT | 20+ | 20+ (COCO, YOLO, VOC, KITTI...) | N | N | Y | MIT; Enterprise from $12,000/yr | `docker compose up` |
| Label Studio | many | many | N | N | Y | Apache-2.0; Cloud/Enterprise paid | Docker or pip |
| Labelme | own JSON | VOC, COCO, JSON | N | N | Y (desktop) | GPL-3.0 | pip |
| X-AnyLabeling | many | COCO, VOC, YOLO, DOTA, MOT, others | N | N | Y (desktop) | GPL-3.0 | pip / binaries (?) |
| FiftyOne | many | many | N (eval and zoo) | N | Y (core) | Apache-2.0; Teams paid | pip |
| LightlyStudio | COCO, YOLO, folders | COCO, YOLO, VOC | N (LightlyTrain separate) | N | Y | Apache-2.0 | pip |
| Roboflow | many | many | Y (hosted) | Y (deploy, Inference) | Inference only | Free / $39+ / Enterprise | None (cloud) |
| Ultralytics Platform | ? | ? | Y (YOLO, cloud GPUs) | Y (deployments) | Enterprise on-prem | Free / $29 seat / Enterprise | None (cloud) |
| Encord / SuperAnnotate / Kili / Labelbox | many | many | P / N | N / P | Enterprise only | Quote | None (cloud) |
| Supervisely | many | many | P (apps) | ? | Enterprise | Free Community / EUR 199+ | Cloud or enterprise |
| SageMaker GT | S3 manifests | manifests | Via SageMaker | Via SageMaker | N | Per object | AWS |
| Autodistill | folders | YOLO and others via target | Y (target models) | N | Y | Apache-2.0 | pip |
| MLflow (adjacent) | n/a | n/a | Tracking only | Registry | Y | Apache-2.0 | pip / container |

Last updated 2026-10-04. Sources: [10-sources.md](10-sources.md); OpenProcessor cells from
README.md, INSTALLATION.md, docs/CURATION.md, LICENSE in this repository.
