# 09 - Gap analysis and positioning

**Summary.** OpenProcessor is useful and differentiated for one audience: teams that want to
build a custom detector from their own images, on their own GPUs, with auto-labeling plus human
confirmation, and keep the model serving in the same stack. It is not a general annotation tool
and should not be positioned as one.

## Is there a real gap?

Searched: open-source annotators, commercial platforms, cloud services, auto-labelers,
curation tools and training platforms (files 02-05). What exists in pieces:

| Capability | Where it exists |
|---|---|
| Auto-label (SAM 3, VLM, open-vocab) | Roboflow, Ultralytics Platform, CVAT, X-AnyLabeling, Autodistill |
| Embedding clusters and sampling | FiftyOne, LightlyStudio, Encord Index |
| Train then deploy | Roboflow, Ultralytics Platform (hosted) |
| Self-hosted, open-source | CVAT, Label Studio, FiftyOne, LightlyStudio |

Not found together in any one self-hosted open-source product: **detector, embedding clusters,
VLM suggestion, human confirm by cluster, train, bake-off, promote back to serving**, with
**class identity by name** and **per-project isolation**. That statement is limited to this
survey of 2026-10-04.

## Differentiators to lean on

1. Closed loop in one stack, self-hosted, data never leaves the host unless a remote VLM is
   explicitly acknowledged.
2. Cluster-first bulk confirmation instead of per-image drawing.
3. Name-based class identity, the lock rule, frozen test holdout, SHA-recorded exports:
   reproducibility and safety features most tools do not document.
4. Versioned prompt packs and region profiles: domain adaptation without code changes.
5. Project isolation enforced at the storage layer; fail-closed behavior.
6. Digest-pinned one-line installer with checksum verification.

## Where competitors are stronger

| Area | Leaders | OpenProcessor today |
|---|---|---|
| Geometry types (polygon edit, keypoints, cuboids) | CVAT, Label Studio, X-AnyLabeling | Boxes first-class; polygons not edited; imported polygon rows become boxes |
| Video and 3D | CVAT, Encord, Segments.ai, Label Studio (limited) | None documented |
| Collaboration, RBAC, SSO, audit | CVAT, Label Studio paid, all commercial | No login in Cropwright; single trusted network |
| Format breadth | CVAT (20+), Datumaro | YOLO, COCO, own; YOLO export only |
| Polish and community | Label Studio (28.4k stars), CVAT (16.9k) | New project |
| Hosted convenience | Roboflow, Ultralytics Platform | Heavy local install |
| Curation metrics maturity | FiftyOne Brain, Cleanlab | Fewer, less proven |
| Managed labeling workforce | Scale, Labelbox, SuperAnnotate | None |
| Training breadth | Roboflow, Ultralytics Platform | YOLO26 detection only |

## Risks

- **License.** AGPL-3.0-or-later plus a vendored AGPL Ultralytics fork limits closed
  commercial embedding; Ultralytics sells an Enterprise License for closed use. SAM 3 is under
  a custom license with gated access. Ultralytics Platform covers a similar story with a
  commercial route. Get legal review for any commercial adopter.
- **Footprint.** GPU required; about 60 GB (core) to about 135 GB (all tiers) disk; 30-60
  minute first install. This excludes laptop and CPU-only users that pip tools serve.
- **No auth.** Documented for trusted LAN only; blocks team and enterprise use.
- **Unproven quality claims.** No published public-data accuracy or time-to-model benchmark;
  performance work (#40) is still open.
- **Crowding.** Roboflow, Ultralytics Platform and CVAT are all adding SAM 3 and training or
  assist features; the differentiator can erode.
- **Dependency risk.** VLM and segmenter models are third-party and change.

## Recommended roadmap priorities (judgement, not research fact)

1. Publish a reproducible public-data benchmark (COCO or Open Images subset): time to a
   trained detector and accuracy, versus manual labeling and versus Autodistill-style
   auto-label-only. This is the single most credible proof.
2. Authentication and basic roles (token or reverse-proxy recipe first, then users).
3. Export breadth: COCO export; polygon and mask preservation in import and export; optional
   segmentation training.
4. Interoperability: converters or direct readers for VOC, CVAT and Label Studio; a Label Studio
   or CVAT bridge so existing annotation tools can be used for geometry edits.
5. Smaller on-ramp: a documented CPU-light or minimal path and a public demo or screencast.
6. Quality metrics: borrow FiftyOne-style mistakenness and Cleanlab-style label-error checks.
7. Video (frame sampling at least) only after the above.

## Positioning and messaging

- Tagline direction: "From raw images to a served detector, self-hosted: clusters and a VLM do
  the bulk labeling, you confirm."
- Say what it is not: not a replacement for CVAT's drawing tools; pair with it when geometry
  work is needed.
- Lead with data control, reproducibility (name-based classes, lock rule, holdout) and the full
  loop; avoid unmeasured speed claims.
- Compare honestly against: Roboflow (hosted equivalent), CVAT (annotation), FiftyOne and
  LightlyStudio (curation), Autodistill (auto-label only).

## Target users

| User | Fit |
|---|---|
| Small team or lab with a GPU and private images (industrial, wildlife, medical-adjacent, security) | Strong |
| Homelab and hobbyist building a custom detector | Good if hardware allows |
| ML team wanting reproducible dataset-to-serving with class-name safety | Strong |
| Annotation vendor or large multi-annotator team | Poor today (no RBAC, boxes only) |
| Teams needing video, 3D or segmentation labels | Poor today |
| Closed-source commercial products embedding it | Needs license review |

Last updated 2026-10-04. Sources: [10-sources.md](10-sources.md) and files 02-08.
