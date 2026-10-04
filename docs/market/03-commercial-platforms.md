# 03 - Commercial platforms

**Summary.** Commercial platforms win on polish, collaboration, hosted convenience and managed
labeling workforces. Most gate self-hosting or VPC to enterprise contracts, and most hide
prices. Pricing below is from vendor pages fetched 2026-10-04 unless marked unverified.

| Product | Status / focus | Published pricing | Self-host | Notes |
|---|---|---|---|---|
| **Roboflow** | Hosted annotate, train, deploy; open Inference server | Free (10 credits/month); Core from $39/month (20 credits, $3.90 each, scaling to 500 credits at $1,399); Enterprise custom | Inference is self-hostable (Apache-2.0 core; enterprise parts need contract; restrictive-license models need commercial add-on). Annotate and training are cloud | Auto Label with SAM 3 or Gemini; Smart Select (SAM) polygons; Teams add-on gives SSO/RBAC; free/Core workspaces can share to public Universe |
| **Ultralytics Platform** | Annotate, train, deploy for YOLO (announced 2026-03-18) | Free $0 (AGPL-3.0 terms); Pro $29/seat/month; Enterprise custom; cloud training per GPU-hour from $0.24 | Enterprise on-prem option | Free plan lists manual, SAM 3.1 and YOLO smart annotation. Direct overlap with train-and-deploy. YOLO26 is the current line |
| **Labelbox** | Annotation plus alignment/RLHF services | Free tier; Starter pay-as-you-go ($0.10 per LBU per aggregator pages, unverified); Enterprise custom. Official pricing URL returned 404 | Not found | Official site positioning not verified; do not assume classic CV is a focus |
| **Encord** | Annotate, Index, Active | Starter, Team, Enterprise; no dollar prices; Starter free status disputed between sources | VPC and on-prem at Enterprise | Closest architecture to curate-then-annotate; pricing page truncated on fetch |
| **SuperAnnotate** | Multimodal editor, curation, orchestration | Starter, Pro, Enterprise by compute hours; no dollar prices | Not stated | SSO at Pro; AI/human routing |
| **V7 (Darwin)** | The V7 site now sells "V7 Go" (document automation); Darwin product page not re-verified | Quote only (for V7 Go) | n/a | Treat Darwin status as unverified |
| **Supervisely** | Image, video, 3D, DICOM | Community free (5 GB, 10,000 files, 30-day retention); Pro from EUR 199/month; Enterprise custom | Enterprise: self-hosted or cloud, offline | Large app ecosystem (unverified detail) |
| **Dataloop** | End-to-end data and pipelines | Quote; aggregator figures conflict ($14/user/year vs tiers) and are unreliable | Multi-cloud, on-prem, hybrid per aggregator pages | Treat all pricing as unverified |
| **Kili** | Multimodal, QA | Free trial (100 assets); Grow (up to 20 seats, 50,000 assets) custom; Enterprise custom | On-prem add-on, Enterprise only | SOC 2 Type II, ISO 27001, HIPAA stated |
| **Segments.ai** | Image, point cloud, sensor fusion | Core from $2.67/hour (3,600 hours/yr minimum); Fusion, Enterprise custom | Not stated | Strong 3D |
| **Scale AI** | Enterprise data engine; Rapid self-serve | Rapid: $0.05 per labeling unit self-labeling after free allocation (via aggregator); enterprise custom | None | Meta invested $14.3B for 49% in June 2025; some frontier-lab customers reduced work (secondary sources); later 2026 corporate events single-sourced, unverified |
| **Hasty** | Acquired by CloudFactory (2022-09-08) | not checked | not checked | Current availability unverified; no shutdown notice found |
| **Snorkel** | Programmatic labeling and expert data services | Quote | Not checked | 2026 layoffs report single-sourced, unverified |
| **SageMaker Ground Truth / Plus** | AWS labeling | Per object reviewed (free tier 500 objects/month for two months); Plus per label by quote | Cloud only | Auto-labeling bills training/inference instances |
| **Azure ML data labeling** | Image labeling with ML-assisted prelabel and clustering | Part of Azure ML | Cloud only | Docs recommend migrating labeling workloads to third-party providers; exports CSV, COCO, AzureML dataset |
| **Vertex AI Data Labeling** | Retired | n/a | n/a | Deprecation date discrepancy noted in file 01 |
| **Amazon Rekognition Custom Labels** | Existing service | not checked | Cloud only | A search for availability changes found no official restriction notice; status unverified |

## Reading the table

- Commercial value is workforce plus polish plus SLA; those are not what a self-hosted loop
  competes on.
- Where commercial tools overlap OpenProcessor most: Roboflow (Auto Label, train, deploy,
  self-hosted inference), Ultralytics Platform (annotate, train, deploy YOLO), Encord (curate
  then annotate), SuperAnnotate (AI/human routing).
- Compared on data control: all of the above require sending images to a vendor cloud unless
  an enterprise on-prem contract exists.

Last updated 2026-10-04. Sources: [10-sources.md](10-sources.md) section B.
