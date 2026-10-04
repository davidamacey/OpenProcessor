# 01 - Landscape overview

**Summary.** The image-labeling market splits into five groups. Open-source annotators and
commercial platforms are crowded. Foundation-model auto-labeling (SAM 3, Grounding DINO,
VLMs) became a standard feature in 2025-2026. Curation and active-learning tools are fewer and
several have been absorbed or retired. Few products own the whole loop through training and
serving, and almost none do it self-hosted and open-source.

## Market map

| Category | Representative products | Typical delivery | Typical license / cost |
|---|---|---|---|
| Open-source annotation | Label Studio, CVAT, Labelme, X-AnyLabeling, Make Sense, Doccano (text), Diffgram | Self-host, desktop, browser | Apache-2.0, MIT, GPL-3.0, custom; free |
| Commercial annotation platforms | Roboflow, Labelbox, Encord, SuperAnnotate, V7, Supervisely, Dataloop, Kili, Segments.ai | SaaS; enterprise on-prem | Seats, credits, units, hours; enterprise quotes |
| Cloud-provider labeling | SageMaker Ground Truth, Azure ML labeling, (Vertex AI Data Labeling retired) | Cloud service | Per object or workforce |
| Auto-labeling / foundation models | Autodistill, Grounded-SAM, SAM 3, Roboflow Auto Label, CVAT agents, VLMs | Library or feature | Mostly open weights; custom licenses |
| Curation / active learning | FiftyOne, LightlyStudio, Cleanlab, Encord Active (archived), Aquarium (CV product retired) | Python SDK plus app | Open core plus paid tiers |
| Training / MLOps adjacent | Ultralytics Platform, Roboflow Train, Edge Impulse, MLflow, NVIDIA Triton | SaaS or self-host | Credits, GPU-hours, open source |

## Trends (all dated; see [10-sources.md](10-sources.md))

- **Foundation-model assist is now standard.** Roboflow Auto Label offers SAM 3 and Gemini;
  CVAT Team plans list SAM 2/3 and Hugging Face/Roboflow integrations; Ultralytics Platform
  lists SAM 3.1 and YOLO "smart" annotation on the free plan; X-AnyLabeling bundles SAM 1-3,
  Grounding DINO, YOLO-World and Qwen3-VL; Labelme lists SAM-family models.
- **SAM 3 (released 2025-11-19)** adds open-vocabulary concept prompts (a short noun phrase
  returns every matching instance), under Meta's custom "SAM License" with gated checkpoints.
- **VLMs with grounding** (for example Qwen3-VL, Apache-2.0) can output boxes, so labeling by
  prompt is feasible without a detector for the class.
- **Consolidation and retreat.** Vertex AI Data Labeling was deprecated and shut down (dates
  in Google's own pages disagree: 2024-07-01 vs 2024-10-03); Microsoft docs advise migrating
  Azure ML labeling workloads to third parties; Encord Active (open source) was archived
  2025-08-07; Aquarium's CV curation product was shut down and the team joined Notion;
  LabelImg (archived 2024-02-29) and VoTT (archived 2021-12-07) are unmaintained; Edge Impulse
  was acquired by Qualcomm (March 2025).
- **Vendors move toward LLM/agent data.** Snorkel (programmatic labeling and expert data
  services) and Scale (defense, robotics; Meta took a 49% stake in 2025) shifted focus; one
  source says Labelbox positions around alignment/RLHF services (unverified for classic CV).
- **Ultralytics launched a hosted Platform (announced 2026-03-18)** covering annotate, train
  and deploy, directly overlapping the "label then train then deploy" story.

## Demand drivers

- Cheap, strong detectors (YOLO family) shift the bottleneck to labeled data for niche domains.
- Auto-labeling makes first-pass labels nearly free; human verification and class decisions
  become the cost.
- Data-residency and privacy needs (medical, industrial, security imagery) favour self-host;
  commercial vendors typically gate on-prem or VPC to enterprise tiers (Encord, Kili,
  Supervisely, Dataloop, SuperAnnotate; see file 03).
- Local GPUs are common among small teams and homelabs, and local VLMs are viable.

## Pricing models seen

| Model | Examples | Notes |
|---|---|---|
| Free open source, paid enterprise | CVAT (Enterprise from $12,000/yr self-hosted), Label Studio, FiftyOne | RBAC, SSO, analytics often in paid tier |
| Credits | Roboflow (Core from $39/month; free 10 credits/month), CVAT AI tool calls | Credits per training and inference |
| Per seat | CVAT Team ($33/user/month, $23 annual), Label Studio Starter Cloud ($99 base + $49/extra user) | |
| Usage units | Labelbox (reported $0.10 per LBU, via aggregator pages; unverified), Segments.ai (from $2.67/hour) | |
| Per object | SageMaker Ground Truth | Plus workforce fees |
| Quote only | Encord, SuperAnnotate, Dataloop, V7, Kili Grow/Enterprise | |
| GPU-hour | Ultralytics Platform (from $0.24/hr per its pricing page) | Plus $29/seat/month Pro |

## Market size (analyst figures, low confidence)

Firms disagree several-fold because scope and base year differ. For the "data annotation
tools" market: Grand View Research about USD 2.1B (2026) to USD 5.3B (2030), CAGR 26.3%;
Mordor Intelligence USD 2.32B (2025) to USD 12.42B (2031), CAGR 32.3%; Technavio about
USD 1.10B (2025), CAGR 28.4%; SkyQuest USD 1.63B (2025) to USD 10.53B (2033). Research
Nester's USD 6.98B (2025) is an outlier. These are paywalled-report press summaries, not
audited, and include services. Use as "roughly low single-digit USD billions, growing
20-30 percent per year, per analysts", nothing more.

Last updated 2026-10-04. Sources: [10-sources.md](10-sources.md) (sections A, B, E).
