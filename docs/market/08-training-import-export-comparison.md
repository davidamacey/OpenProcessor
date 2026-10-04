# 08 - Training, import and export comparison

**Summary.** Few labeling tools train; those that do are hosted (Roboflow, Ultralytics
Platform). OpenProcessor trains YOLO26 locally and promotes to Triton. Its format coverage is
the narrowest of the field: YOLO, COCO and its own export in, YOLO out.

## Training

| Product | Training | Notes |
|---|---|---|
| **OpenProcessor** | YOLO26 through a separate trainer service (multi-size campaigns, bake-off, frozen holdout, lineage manifest, MLflow); needs about 16 GB free VRAM while training per README tiers | Promote exports to Triton under the project; a subset or single-class run without its `class_remap` is refused with 422 (docs/CURATION.md) |
| Roboflow | Hosted training; credits (about 30 trainings on the free 10 credits per vendor page) | |
| Ultralytics Platform | Cloud GPUs from $0.24/hr; example 1,000 images YOLO26n 100 epochs about $6 (vendor page) | AGPL-3.0 terms on Free; Enterprise License for closed use |
| Autodistill | Target models YOLOv8, YOLO-NAS, YOLOv5, DETR, ViT | Library only |
| Edge Impulse | Developer plan free (60-minute jobs, 3 private projects) per search summary | Edge-device focus; Qualcomm-owned since 2025 |
| CVAT, Label Studio, Labelme | None | Export then train elsewhere |
| LightlyTrain | Self-supervised pretraining library | Separate from LightlyStudio |
| MLflow | Tracking and registry, not training | Apache-2.0; OpenProcessor's trainer uses it |

## Import formats

| Product | Formats |
|---|---|
| OpenProcessor | YOLO (`data.yaml`), COCO, OpenProcessor export; VOC, CVAT, LabelMe, Label Studio via converters (FiftyOne, supervision); polygon, OBB, keypoint rows kept as boxes only |
| CVAT | 20+ (COCO, YOLO, VOC, KITTI) |
| Label Studio | Many (not enumerated here) |
| Datumaro | CIFAR, COCO, CVAT, ImageNet, KITTI, LabelMe, MNIST, Open Images, VOC, TF Detection API, YOLO and more |
| supervision | YOLO, VOC, COCO |
| FiftyOne | COCO, VOC, ImageNet and more |
| X-AnyLabeling | COCO, VOC, YOLO, DOTA, MOT, MASK, PPOCR, MMGD, VLM-R1, ShareGPT |
| LightlyStudio | COCO, YOLO, folders |
| Azure ML labeling | Export CSV, COCO, AzureML dataset |

## Export formats

| Product | Formats |
|---|---|
| OpenProcessor | YOLO multi-class; single-class or class subset; crop mode; manifest with `dataset_sha`; splits by source image; holdout images forced to test |
| CVAT | 20+ |
| Labelme | VOC, COCO (instance), JSON |
| Make Sense | YOLO, VOC XML, COCO JSON, VGG JSON, CSV |
| LightlyStudio | COCO, YOLO, VOC |
| X-AnyLabeling | COCO, VOC, YOLO, DOTA, MOT, others |

## Takeaway

OpenProcessor's name-based class identity and holdout-aware, SHA-recorded export are stronger
reproducibility features than most open-source tools document, but the lack of COCO export,
polygons and segmentation or pose training limits who can use it today.

Last updated 2026-10-04. Sources: [10-sources.md](10-sources.md); docs/CURATION.md, README.md.
