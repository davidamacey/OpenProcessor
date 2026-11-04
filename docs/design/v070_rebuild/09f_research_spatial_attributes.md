# Research plan: spatial scenes and attribute signals for the clean rebuild

| Field | Value |
|---|---|
| Superseded parts | the landing lanes, waves and file ownership of this plan's section 8, its GPU lock and its GPU-hour estimates are superseded by the program (review RR-16, owner answers of 2026-11-04): lanes in [program/03b](03d_stage4_job_families.md) (W2.3a and W2.3b), [program/03d](03b_stage2_control_data.md), [program/03c](03f_stage6_scheduler_scaleout.md), [program/05](05_b4_stateless_api.md) and [program/03g](06e_w5e_plate_text.md); one lock `gpu<N>.lock` per card shared by every topic ([program/00b](reference/traceability.md) X-14); short tiered runs under a cap of about 10 GPU hours before the tag ([program/00f_research_track.md](08_research_prototype_track.md) section 2.1). A fresh agent loads those files first |
| Status | plan, research-first. Written 2026-11-04 on `research/spatial` from `origin/main` at `49731595`. **Nothing was run for this plan.** The only measured fact it relies on is the EXIF census of 2026-11-04 in section 2.3 |
| Issues | #348 (spatial scene discovery and photogrammetry, parked), #400 (attribute-signal layer, parked); #346 read for how attributes feed clustering |
| Replaces | the per-issue plans `issue-348-plan.md` and `issue-400-plan.md` (deleted in this commit; their content is carried here and their gists stay the issue announcements) |
| Reads with | [program/00a_decision_register.md](reference/decision_register.md) (section 2.1, decisions section 6), [program/00_master_plan.md](00_master_plan.md), [program/02_control_plane_database.md](02_control_plane_database.md), [program/02a_postgres_schema.md](02a_postgres_schema.md), [program/02b_jobs_locking_events.md](02b_jobs_locking_events.md), [program/02c_migration_ops_tests.md](02c_migration_ops_tests.md), [09b_research_clustering.md](09b_research_clustering.md), [09h_research_plate_text.md](09h_research_plate_text.md), [the spatial research note](../../research/OpenProcessor_Spatial_Memory_Photogrammetry_Research_README.md), [spatial_datasets.md](../../research/spatial_datasets.md), [cluster_types.md](../../research/cluster_types.md) |
| Audience | a fresh agent with no memory of this session: the integration agent (section 11), a coding lane (section 8), a research session (section 6) |
| Owner direction | 2026-11-04: a clean backend rewrite; the research "will materially change how we use the spatial attributes", so research it to best practice first; prototypes are local-venv GPU scripts, no Docker builds; plan first, run nothing |

Plain-words rule (D-45): names first, codes after. "Set P" is the owner's private high-resolution
photo collection; it is described only by numbers.

---

## 1. Plain summary

**What this is about.** Photos carry facts about where and when they were taken (EXIF capture
time, camera, sometimes GPS), and boxes inside them have attributes (colour, which side of the object
is visible). Several photos may show the same physical place, and some such groups can be turned
into a 3D model with camera positions, which places photos from different years on one timeline.

**What exists today.** Almost nothing. Ingest reads EXIF only for orientation. No capture time,
camera or GPS is stored. Colour exists only as a VLM prompt-pack suggestion. No view tag, no place
grouping, no reconstruction. Nothing spatial has been measured on our hardware.

**What the literature and industry say** (section 3):
- Capture metadata is a solved standard: EXIF 3.0 and its time-offset tags, with XMP and IPTC as
  fallbacks. Immich, the reference self-hosted photo app, reads it with ExifTool and keeps GPS
  behind one visibility rule.
- Same-place discovery is always "retrieve, then verify geometrically": a place-recognition vector
  proposes pairs, local features plus robust two-view geometry confirm them. COLMAP 4.2 (with the
  former GLOMAP as its global mapper) reconstructs verified groups. It does not discover them.
- Colour naming and viewpoint have long, honest literatures. Both are cheap to compute but fragile
  (lighting, symmetric objects), so they need a measured bar before anything relies on them.

**What I recommend.**
1. **Ship one thing in the v0.7.0 rebuild: capture metadata at ingest.** Capture time with its
   offset and source, camera make and model, lens and focal length, and a `has_location` flag.
   Store no coordinates in v0.7.0 (section 4, D1). This is the only signal that can be lost by
   waiting: the original bytes are only guaranteed at ingest. It is standard, cheap (the header is
   already open) and has a reader on day one (a capture-time filter and the item wire).
2. **Keep every other door open, with no dead code.** The rebuild already leaves the right doors:
   `jobs.kind` has no CHECK, the images index is k-NN enabled, project delete removes the whole
   storage prefix, and RLS has a parent-table helper. This plan fixes the exact names and types for
   scenes, reconstructions and attribute signals (section 5), so a later Alembic revision adds them
   without redesign. They are **not** created in v0.7.0.
3. **Colour and view: production-grade methods, but measure before shipping.** Both come from data
   we already store (the PE crop vector and its text tower) or from crops. They can be backfilled at
   any time with no data loss. They land after the tag, once a CPU-only prototype passes its bar
   (P-2, P-3), as `item_scores` scorers with fixed per-box fields.
4. **Place grouping and photogrammetry stay research.** Run #348's registered S1 experiment (four
   stages, public data, pycolmap and MegaLoc). Run it after the tag in a research window. Product
   work starts only on a GO (section 6, P-4 to P-7).
5. **Privacy: location is the most sensitive thing this plan touches.** The rules are neutral and
   fail closed (section 9). Set P location data never leaves the machine.

---

## 2. What exists and what was measured

### 2.1 Code today (verified at `49731595`)

| Capability | State | Where |
|---|---|---|
| EXIF read at ingest | orientation only (`header.getexif().get(EXIF_ORIENTATION, 1)`) | `src/services/curation/image_decode.py` `_decode` |
| Capture time, offset, camera, lens, GPS | **not read, not stored anywhere** (`rg -n -i "gps\|DateTimeOriginal\|capture_time" src` finds none) | - |
| Images index | `width`, `height`, `imohash`, `phash`, `indexed_at`, `embedding`, `pe_embedding`; `index.knn` true (a final setting) | `src/clients/curation_opensearch/bodies_core.py` `_images_body` |
| Region box element | fixed keys `box_id`, `bbox_norm`, `state`, `score`, `detector*`, `text*`, `cluster_*`, `detected_at`; no attribute keys | `src/domain/region_boxes.py` `RegionBox` |
| Colour | VLM pack attribute (11-value closed vocabulary) in `assist.attributes[]`, a suggestion store no export, training or holdout path reads | `examples/prompt_packs/general_objects.json`, `src/clients/curation_opensearch/assist_fields.py` |
| View tag | none; the PE text tower exists (`PeEncoder.encode_text`) | `src/clients/pe_encoder.py` |
| Near-duplicate groups | `dup_group_id` per crop; not read by clustering or the loop gate | `src/services/curation/item_scores/near_dup.py` |
| Scorer pattern for per-crop signals | `CropScorer` protocol and registry, job family `item_scores` | `src/services/curation/item_scores/` |
| Place grouping, pair evidence, scenes, reconstruction | none | - |
| Spatial research tooling | the dataset fetcher and verifier only | `scripts/datasets/fetch_spatial_sets.py` |

**Found while reading (correctness, not research).** `src/services/curation/export_support.py`
`_copy_or_resize_one`:
- the resize and `crop_norm` path opens with `Image.open(...).convert('RGB')` and **no**
  `ImageOps.exif_transpose`. For EXIF orientations 2 to 8 the exported pixels are in sensor
  orientation, while the labels are normalized against the oriented frame that ingest decoded. For
  tags 5 to 8 even the width and height swap, so labels and crops land in the wrong place. The
  5b-D1 caller list in [program/06b_b5_decode_once.md](06b_b5_decode_once.md) does not name
  this file;
- the no-resize path is `shutil.copyfile`, so exported originals keep their embedded EXIF,
  **GPS included**. That is a location leak today that is independent of any field we store.

Both go to the integration agent (section 11).

### 2.2 The control plane this builds on (planned, not merged)

From [program/02a_postgres_schema.md](02a_postgres_schema.md) and
[program/02b_jobs_locking_events.md](02b_jobs_locking_events.md):
- `uuidv7()` keys, `text` plus `CHECK` for closed sets, `jsonb` only for pydantic-validated
  bodies, `storage_key` relative keys;
- RLS `FORCE`d on every `project_id` table, plus `op_scope_allows(pid)` for child tables;
- `jobs.kind` validated in code, not by a CHECK, so a new kind needs no migration;
- queues with a consumer test (`tests/test_queue_consumers_from_compose.py`) and the GPU-queue
  pattern (`gpu_train`, `gpu_export`, `gpu_eval`: prefork 1, a profile, GPU from `.env`);
- the storage seam (D-46) with roots `source`, `projects`, `models`, `cache`, `upgrade`; project
  delete removes `projects/<slug>/` as a prefix;
- revisioned config documents with a closed `kind` CHECK (`prompt_pack`, `region_profile`,
  `open_vocab_set`, `ingest_profile`, `vlm_endpoint`) and per-project activation axes;
- `project_settings` with OCC on every write; the outbox `search_projection_tasks` for search
  projections owned by Postgres (D-47).

### 2.3 Measured so far

| What | Result | Source |
|---|---|---|
| EXIF on local public samples (Pillow `getexif`, `get_ifd`), 2026-11-04 | COCO val (2,000 files): 0 with any EXIF. Open Images plate pin from the S3 mirror (300 files): 1 with capture time, GPS and camera | the former #400 plan, section 2.4 |
| Spatial public datasets fetched and integrity-checked, 2026-10-29 | 10 sets, 158 GB, 67 archives, 0 failures after a refetch | [spatial_datasets.md](../../research/spatial_datasets.md) |
| Place retrieval, pair verification, grouping, reconstruction | **not run** | - |
| Colour or view accuracy | **not run** | - |
| EXIF coverage on set P | **not run** (owner-run, P-1) | - |

The census means COCO and the Open Images mirror cannot back any EXIF, burst or GPS claim. The
public sets that can are listed in section 6.3.

---

## 3. Literature and industry review

Each row cites the primary source. Vendor and README claims are labelled as such. Licences were read
on 2026-11-04 and must be re-checked before anything ships.

### 3.1 Capture metadata standards and libraries

| Source | What it settles |
|---|---|
| EXIF 3.0, CIPA DC-008-2023 / JEITA CP-3451F; corrected translation 2024; Exif 3.1 published 2026-01-30 ([CIPA history](https://www.cipa.jp/e/std/history_sec.html), [Library of Congress format record](https://www.loc.gov/preservation/digital/formats/fdd/fdd000618.shtml)) | Tag meanings. `DateTimeOriginal` (0x9003) is capture time in local wall-clock time **without a zone**; `SubSecTimeOriginal` (0x9291); `OffsetTimeOriginal` (0x9011, since 2.31) is the UTC offset; `DateTimeDigitized` (0x9004) is the digitising time; IFD0 `DateTime` (0x0132) is the **modification** time, never capture time. The GPS IFD holds `GPSLatitude`/`Ref`, `GPSLongitude`/`Ref`, `GPSAltitude`/`Ref`, `GPSTimeStamp` plus `GPSDateStamp` (UTC), `GPSImgDirection`/`Ref` (camera heading), `GPSHPositioningError` (metres, since 2.31), `GPSDOP`, `GPSMapDatum`. `BodySerialNumber` (0xA431) is a device identifier. Exif 3.0 adds a UTF-8 type for free-text tags |
| IPTC Photo Metadata mapping guidelines ([IPTC](https://iptc.org/std/photometadata/documentation/mappingguidelines/)) | How EXIF, XMP and IPTC fields map: IPTC "Date Created" = `DateTimeOriginal` + `SubSecTimeOriginal` + `OffsetTimeOriginal`. XMP (ISO 16684-1) `exif:DateTimeOriginal`, `xmp:CreateDate` and `photoshop:DateCreated` are the fallbacks when an editor dropped EXIF |
| Immich, reading order ([XMP sidecars](https://docs.immich.app/features/xmp-sidecars/)) | Prioritized date tags: `SubSecDateTimeOriginal`, `DateTimeOriginal`, `SubSecCreateDate`, `CreateDate`, `CreationDate`, `MediaCreateDate`, ..., then file times. Time zones come from ExifTool through exiftool-vendored, whose documented heuristics include the offset tags and the GPS UTC time ([exiftool-vendored dates](https://photostructure.github.io/exiftool-vendored.js/index.html#dates)) |
| Immich, reverse geocoding ([docs](https://docs.immich.app/features/reverse-geocoding/)) | Offline: GeoNames data loaded into Postgres; city, state and country per asset; no external call |
| Immich, location leak fixed 2026-10 ([PR #32561](https://github.com/immich-app/immich/pull/32561)) | A new search path served full EXIF, GPS included, through shared links whose "show metadata" was off. The fix moved the check into one asset mapper that every path uses. **Lesson: one visibility helper for location on every read path, plus a test that enumerates the paths** |
| Pillow 13 (pinned `==13.0.0` in [program/01_package_pins.md](01_package_pins.md)) ([ExifTags](https://pillow.readthedocs.io/en/stable/reference/ExifTags.html), [`Image.Exif`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Exif)) | `getexif()` plus `get_ifd(ExifTags.IFD.Exif)` and `get_ifd(ExifTags.IFD.GPSInfo)` read every tag above from the header ingest already opens. Parsing XMP with `getxmp()` needs `defusedxml`, which the stack does not have |
| ExifTool ([exiftool.org](https://exiftool.org/)), the Perl licence | The de facto reference reader, including maker notes and odd formats. It writes losslessly too (it can strip GPS from a JPEG without re-encoding). Through [PyExifTool 0.5.6](https://pypi.org/project/pyexiftool/) (GPL-3.0-or-later or BSD; last release 2023-10) it is a subprocess plus a system dependency |
| ExifRead 3.5.1 ([PyPI](https://pypi.org/project/ExifRead/)), BSD-3 | Pure Python, maintained (2025 releases), reads HEIC too. A fallback candidate if P-1 shows Pillow misses formats we ingest |
| timezonefinder ([repo](https://github.com/jannikmi/timezonefinder)), MIT | Offline IANA zone from coordinates. That is a derived location; see the rules in section 9 |

### 3.2 Place recognition (retrieval)

| Method | Source | Notes for us |
|---|---|---|
| NetVLAD | Arandjelović et al., CVPR 2016, [arXiv:1511.07247](https://arxiv.org/abs/1511.07247) | The learned-aggregation baseline; hloc ships it |
| CosPlace | Berton et al., CVPR 2022, [arXiv:2204.02287](https://arxiv.org/abs/2204.02287) | Classification-trained VPR; the same group's benchmark: [arXiv:2204.03444](https://arxiv.org/abs/2204.03444) |
| MixVPR | Ali-bey et al., WACV 2023, [arXiv:2303.02190](https://arxiv.org/abs/2303.02190) | Feature-mixing aggregation (ResNet-50, GSV-Cities) |
| EigenPlaces | Berton et al., ICCV 2023, [arXiv:2308.10832](https://arxiv.org/abs/2308.10832) | Trained for viewpoint robustness: the property a photo library needs |
| AnyLoc | Keetha et al., RA-L 2024, [arXiv:2308.00688](https://arxiv.org/abs/2308.00688) | Training-free DINOv2 + VLAD. The authors report large gains in unstructured scenes. The best setting is ViT-G with 49,152-d descriptors, too large to store per image |
| SALAD | Izquierdo and Civera, CVPR 2024, [arXiv:2311.15937](https://arxiv.org/abs/2311.15937) | DINOv2 + optimal-transport aggregation. Code GPL-3.0; **the weights carry no stated licence**: measurement-only |
| MegaLoc | Berton and Masone, CVPRW 2025, [arXiv:2502.17237](https://arxiv.org/abs/2502.17237), [repo](https://github.com/gmberton/MegaLoc) (MIT) | DINOv2-base + SALAD-style head trained on five datasets. 8,448-d L2-normalized, 322x322 input, one-line `torch.hub` load. The authors claim state of the art on most VPR sets (to be measured). The README notes the training data cannot be redistributed; no separate weights terms are stated. **The shippable challenger** |
| PE-Core-L frame vector (stored) and MobileCLIP (core) | in the stack | Semantic, not place, vectors. The G-2 reuse rule (section 6) decides whether a place vector earns its cost |

### 3.3 Local features, matching and verification

| Method | Source | Notes |
|---|---|---|
| SuperPoint | DeTone et al., CVPRW 2018, [arXiv:1712.07629](https://arxiv.org/abs/1712.07629) | **Restrictive licence on weights and inference file** (LightGlue README). Excluded |
| ALIKED | Zhao et al., IEEE TIM 2023, [arXiv:2304.03608](https://arxiv.org/abs/2304.03608) | BSD-3 (per the LightGlue README); built into COLMAP 4.x |
| LightGlue | Lindenberger et al., ICCV 2023, [arXiv:2306.13643](https://arxiv.org/abs/2306.13643), [repo](https://github.com/cvg/LightGlue) | Apache-2.0 code and weights; weights for SuperPoint, DISK, ALIKED and SIFT. COLMAP 4.x exposes SIFT and ALIKED LightGlue matchers |
| hloc | Sarlin et al., CVPR 2019, [arXiv:1812.03506](https://arxiv.org/abs/1812.03506), [repo](https://github.com/cvg/Hierarchical-Localization) (Apache-2.0) | Retrieval, matching and localization on pycolmap. Not needed for discovery; the tool for **registering old photos to a reference scene** (stage S3) |
| Doppelgangers | Cai et al., ICCV 2023, [arXiv:2309.02420](https://arxiv.org/abs/2309.02420) | A learned classifier for "similar but different" structures. The documented fix for repeated-structure false edges. Licence to check; an arm only if G-3 fails |
| IMC 2020 protocol | Jin et al., IJCV 2021, [arXiv:2003.01587](https://arxiv.org/abs/2003.01587) | mAA of relative pose at angular thresholds: the metric S1d uses |

### 3.4 Structure from motion

| Engine | Source | Notes |
|---|---|---|
| COLMAP incremental SfM | Schönberger and Frahm, CVPR 2016 ([paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html)); [docs](https://colmap.github.io/) | The baseline, BSD |
| Global SfM (GLOMAP) | Pan et al., ECCV 2024, [arXiv:2407.20219](https://arxiv.org/abs/2407.20219) | The standalone repo is archived; it lives on as COLMAP's global mapper |
| COLMAP 4.1 to 4.2.1 ([releases](https://github.com/colmap/colmap/releases)) | release notes | 4.1.0: GPU bundle adjustment (Caspar), ALIKED fixes. 4.2.0: LoMa ONNX features; the global mapper reconstructs **every connected component** (`GlobalMapper.multiple_models`). 4.2.1 (2026-09-29): ONNX Runtime leak fix and restored legacy BA in the global mapper. An AnyCalib ONNX backend (Apache-2.0) exists for calibration |
| pycolmap ([docs](https://colmap.github.io/pycolmap/index.html)) | docs | `pip install pycolmap` (CPU); `pycolmap-cuda12` 4.2.1, Linux x86-64, CPython 3.10-3.14 ([PyPI](https://pypi.org/project/pycolmap-cuda12/)). `extract_features`, `match_exhaustive`, `match_spatial` (GPS priors), `match_vocabtree`, `match_image_pairs`, `geometric_verification`, `incremental_mapping`, `global_mapping` |
| Geo-registration | [COLMAP FAQ](https://colmap.github.io/faq.html) (`model_aligner`) | A similarity transform from scene-local to geographic coordinates, given GPS or ground-control positions. SfM alone has unknown scale |

### 3.5 Camera pose from one image (no reconstruction)

| Method | Source | Notes |
|---|---|---|
| EXIF focal length | EXIF `FocalLengthIn35mmFilm` | Gives the field of view when present; the cheapest intrinsics prior |
| GeoCalib | Veicht et al., ECCV 2024, [arXiv:2409.06704](https://arxiv.org/abs/2409.06704), [repo](https://github.com/cvg/GeoCalib) | Gravity (roll, pitch) and focal length from one image, with uncertainty, about 100 ms per image (authors). Licence to check |
| AnyCalib | [arXiv:2503.12701](https://arxiv.org/abs/2503.12701) | Model-agnostic single-view calibration. Shipped as an Apache-2.0 ONNX backend for COLMAP |
| EXIF heading | `GPSImgDirection` | Compass heading of the camera, when the phone wrote it. Noisy; a prior, never truth |

### 3.6 Colour naming

| Source | What it says |
|---|---|
| Berlin and Kay, *Basic Color Terms*, 1969 | The 11 English basic terms: black, white, red, green, yellow, blue, brown, orange, pink, purple, grey |
| van de Weijer et al., "Learning Color Names for Real-World Applications", IEEE TIP 2009, [doi:10.1109/TIP.2009.2019809](https://doi.org/10.1109/TIP.2009.2019809); [project page](https://lear.inrialpes.fr/people/vandeweijer/color_names.html) | Colour names learned from real images beat names learned from colour chips. The published sRGB-to-11-terms lookup table has **no stated licence**: measurement-only |
| CIEDE2000, Sharma, Wu and Dalal, Color Res. Appl. 2005 | The perceptual colour difference. scikit-image `rgb2lab` and `deltaE_ciede2000` implement it ([docs](https://scikit-image.org/docs/stable/api/skimage.color.html)) |
| Pillow `Image.quantize` (median cut) ([docs](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.quantize)) | Palette extraction on the crop already decoded |
| CLIP zero-shot (Radford et al. 2021, [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)) | "a photo of a red {class}" against the stored crop vector: no pixels, no new model |
| Known failure | Illumination: the same object changes name under shade, sodium light or white-balance error. Colour constancy is unsolved in general, so flip rate is the metric that matters (the former #400 plan, 4.4) |

### 3.7 Viewpoint estimation

| Source | What it gives |
|---|---|
| Pascal3D+ (Xiang, Mottaghi and Savarese, WACV 2014, [project](https://cvgl.stanford.edu/projects/pascal3d.html)) | Azimuth, elevation and in-plane rotation for 12 rigid classes, cars included, on PASCAL VOC and ImageNet images. Public viewpoint ground truth, no person labelling needed |
| Viewpoints and Keypoints (Tulsiani and Malik, CVPR 2015, [arXiv:1411.6067](https://arxiv.org/abs/1411.6067)); Render for CNN (Su et al., ICCV 2015, [arXiv:1505.05641](https://arxiv.org/abs/1505.05641)) | The standard discretized-azimuth formulation and its accuracy metrics |
| CompCars (Yang et al., CVPR 2015, [arXiv:1506.08959](https://arxiv.org/abs/1506.08959)) | Five vehicle views (front, rear, side, front-side, rear-side). Research licence: local cross-check only |
| Orient Anything (Wang et al., ICML 2025, [arXiv:2412.18605](https://arxiv.org/abs/2412.18605), [repo](https://github.com/SpatialVision/Orient-Anything)) | Zero-shot object orientation from a DINOv2 backbone trained on renders. **Licence not found**: measurement-only until it is |
| Zero-shot prompts on the stored PE vector | The cheapest arm (the former #400 plan, 4.3): one text encode per prompt per project plus a dot product |

### 3.8 Products and what they teach

| Product | Lesson |
|---|---|
| Immich ([repo](https://github.com/immich-app/immich)) | Capture metadata, map, offline places, timeline; no 3D. Location shown through one helper. Reverse geocoding is local |
| LibrePhotos, WebODM, Meshroom, OOOSplat (the spatial note, section 4) | Durable photogrammetry jobs exist (WebODM). No self-hosted product does "library to discovered scenes to 3D to dated timeline" unattended. That integration is the gap the spatial note names |

---

## 4. Candidate designs

Legend: ✅ recommended; ❌ rejected; 🟡 conditional (needs a measured bar or an owner decision).

### 4.1 Production-ready methods (standard practice; land on our stack with a bar)

| ID | Design | Verdict | Why |
|---|---|---|---|
| D1 | **Capture metadata at ingest, in v0.7.0**: time (local, offset, UTC when known, source, precision), camera make, model, lens, focal length (mm and 35 mm equivalent), `has_location`, `exif_status`. Read with Pillow from the header ingest already opens; a pure parser in `src/domain/` | ✅ | Only this signal is lost by waiting (the source bytes may be gone later). The standard is settled (3.1). Cost is one header parse. It has readers in v0.7.0 (item wire, a capture-time filter) |
| D1a | Store GPS coordinates in v0.7.0 behind a per-project opt-in | 🟡 (J2) | Nothing in v0.7.0 reads coordinates (no map, no scenes). Data minimisation says a stored location nobody reads is a liability. `has_location` keeps the coverage statistic. Coordinates come with the timeline or scenes, via a header backfill while the source exists |
| D1b | Capture through ExifTool (pyexiftool) in the product | ❌ for ingest | A subprocess plus a Perl system dependency in the API image, per image, for tags Pillow already reads. ✅ as the **research oracle** in P-1: Pillow's output is measured against ExifTool's |
| D1c | XMP date fallback (`defusedxml` + Pillow `getxmp`) | 🟡 (P-1) | Add only if P-1 shows a material share of images with an XMP date and no EXIF date |
| D1d | Camera body serial | ❌ raw; 🟡 keyed hash (J3) | A device identifier. Bursts can use make plus model until #401's secret mechanism exists |
| D1e | File modification time as capture time | ❌ | Not capture time (EXIF spec; Immich treats it as the last fallback, with known wrong results). We store "unknown" instead of a wrong time |
| D1f | Time-zone inference from coordinates (timezonefinder) | 🟡 | Only when the location policy stores location; otherwise the offset comes only from the offset tags or the GPS UTC delta |
| D2 | **Colour per box, three arms**: (a) PE zero-shot over the stored vector; (b) pixel Lab palette on the decoded crop (mask when the region stage has one), named by nearest CIE Lab centroid of the 11 basic terms with CIEDE2000; (c) the existing VLM pack attribute | 🟡 (P-3) | Methods are standard. Accuracy and flip rate on our data are unknown. Arm (a) needs no pixels, so it can backfill everywhere |
| D2x | A trained colour model | ❌ | No case until the cheap arms fail |
| D3 | **View per box**: PE zero-shot prompts per coarse class, calibrated by isotonic regression, abstaining under a margin | 🟡 (P-2) | Bar from the former #400 plan: at least 0.8 per bin, at least 60 per bin. Pascal3D+ gives public ground truth |
| D3a | Orient Anything or another learned orientation model | 🟡 measurement-only | Licence not found |
| D3x | View as a clustering veto | ❌ | Views differ inside one class and one object (the former #400 plan, principle and 4.3) |
| D4 | **Attribute storage: fixed typed keys per signal** (`attr_colour`, `attr_colour_confidence`, `attr_colour_signal`, `attr_colour_calibration`; the same for `view`) on items and on region-box elements | ✅ | Matches the fixed-key rule for box elements, stays greppable, filters without nested queries, and the wire lists fields explicitly (the Immich lesson) |
| D4x | A generic nested `attributes[]` list (key, value, confidence) for measured signals | ❌ | Every filter becomes a nested query. A serializer that loops over keys is how location leaks. Pack-defined VLM attributes keep `assist.attributes[]` (suggestions) |
| D5 | **Signal definitions and versions in Postgres** (`attribute_signals`, `attribute_signal_versions`, `project_attribute_calibrations`); items carry `signal@version` stamps | ✅ (with D2/D3, after the tag) | They have a lifecycle and a revision, so Postgres owns them (rule 1 of 02). Stamps cannot drift, the same pattern as `region_profile@revision` |
| D6 | **Geo-photo timeline, level 1** (capture time plus coarse location: OpenSearch `date_histogram` and `geotile_grid`) | 🟡 (J13) | Immich parity. Needs D1a. A v0.8.0 feature, not a v0.7.0 seam |

### 4.2 Research (no product code before a GO)

| ID | Design | Verdict | Why |
|---|---|---|---|
| R1 | Run COLMAP over the whole library | ❌ | All-pairs work; repeated structure merges places; the global mapper splits components but does not discover them (the former #348 plan, section 1) |
| R2 | Same place = clustering of the existing vectors | ❌ | Place identity is geometric. The stored vectors are semantic. Encoder-limited even for classes (`cluster_program_outcome.md` section 1) |
| R3 | **Retrieve top-K (PE, MobileCLIP, MegaLoc), verify with pycolmap two-view geometry, group verified edges (components and 2-core), reconstruct the discovered group with `pycolmap.global_mapping`** | ✅ research (S1) | The documented pattern of hloc, GLUEMAP and the IMC protocol. One geometry dependency for verification and mapping |
| R4 | GPS and time as candidate priors (radius, sessions) before vision | 🟡 (P-8) | Standard (COLMAP `match_spatial`), but public sets with both images and GPS are few (6.3). Never a replacement for verification |
| R5 | A place vector stored per image (`place_embedding`) | 🟡 (G-2) | Only if MegaLoc beats PE by more than 0.05 recall@20. At 8,448-d about 33 KB per image, so a reduced dimension or an offline FAISS index is part of the decision |
| R6 | Reconstruction readiness report (connectivity, verified share, baseline, registration preview) | ✅ after GO | The spatial note 9.5. A report, never an opaque percentage |
| R7 | Old-photo registration to a reference scene (hloc localization) | 🟡 S3 | Only after GO plus an S1e signal |
| R8 | Dense MVS, splats, viewer | 🟡 (J12) | After GO. COLMAP dense plus gsplat (Apache-2.0) first; OpenMVS (AGPL) only if mesh quality needs it |
| R9 | Single-image gravity and FoV (GeoCalib, AnyCalib) as a pose prior | 🟡 measurement-only | Useful for viewers and verification priors. Licence check for GeoCalib |

---

## 5. Seams the v0.7.0 rebuild must include

Three kinds of seam:
- **(A) ships in v0.7.0**, with a writer and a reader;
- **(B) a door the rebuild must not close.** Already true in the program plans; this plan adds a
  test so it stays true;
- **(C) reserved names and types**, created by a later Alembic revision or mapping migration when
  their research passes. They are not created in v0.7.0, because there is no dead code.

### 5.1 (A) Capture metadata, v0.7.0

**Postgres.** The owner chose to store coordinates now (2026-11-04, program RD-51 reversed; see
[program/02a](02a_postgres_schema.md) section 4), so `project_settings` in
`0001_control_plane` has:

```sql
location_policy  text NOT NULL DEFAULT 'store'   -- owner 2026-11-04: RD-51 reversed, stored now
                 CHECK (location_policy IN ('drop','store_coarse','store'))
```

The 0.6.0 upgrade writes `drop` for every migrated project (fail closed).

**OpenSearch, images index** (a mapping module `src/clients/curation_opensearch/capture_fields.py`
on the `assist_fields.py` pattern, applied through the mapping-migration bootstrap of lane 2F):

| Field | Type | Meaning |
|---|---|---|
| `capture.taken_local` | `date` (`strict_date_hour_minute_second_fraction\|\|strict_date_hour_minute_second`) | wall-clock capture time with no zone, stored as "floating" (as if UTC); for local-hour and day views only |
| `capture.taken_utc` | `date` | set only when the offset is known (offset tag or GPS UTC delta) or the time came from GPS UTC |
| `capture.offset_minutes` | `short` | UTC offset; null when unknown |
| `capture.time_source` | `keyword` | `exif_original`, `exif_digitized`, `gps_utc`, `none` (`xmp` added only if D1c passes) |
| `capture.offset_source` | `keyword` | `exif_offset`, `gps_utc_delta`, `none` |
| `capture.time_precision` | `keyword` | `subsecond`, `second` |
| `capture.camera_make`, `capture.camera_model`, `capture.lens_model` | `keyword` | trimmed, at most 128 characters |
| `capture.focal_length_mm`, `capture.focal_length_35mm` | `float` | from `FocalLength`, `FocalLengthIn35mmFilm` |
| `capture.has_location` | `boolean` | a valid GPS fix was present in the file (coordinates **not** stored unless the policy allows) |
| `capture.exif_status` | `keyword` | `ok`, `absent`, `malformed` (one of the exceptions Pillow raises on malformed EXIF, caught by name) |
| `capture.parser` | `keyword` | `capture@1`, the parser version stamp |

**OpenSearch, items index** (immutable copies, so crop searches filter without a join; like the
provenance stamps they cannot drift): `capture_taken_local` (`date`), `capture_taken_utc`
(`date`). **No location field is ever copied to items.**

**Reserved for D1a (C), images only:** `capture.location` (`geo_point`), `capture.altitude_m`,
`capture.location_accuracy_m`, `capture.heading_deg` (`float`), `capture.location_precision`
(`keyword`: `exact`, `coarse`).

**Code (pure, testable):**
- `src/domain/capture_metadata.py`: `CaptureMetadata` (frozen dataclass) and
  `parse_capture(exif_ifd0, exif_ifd, gps_ifd) -> CaptureMetadata`. It takes plain dicts, so no
  Pillow import, which keeps the domain layering rule. It parses location into the dataclass; the
  writer drops it unless the policy allows. That is tested both ways.
- The extraction (`getexif`, `get_ifd`) in the decode helper; `DecodedImage` gains
  `capture: CaptureMetadata`. 5b-D1 later moves the helper to `src/utils/image_decode.py`, and the
  header read must move with it.
- Rules (each a red/green test):
  - IFD0 `DateTime` is never capture time;
  - an offset is used only from `OffsetTimeOriginal` (or `OffsetTimeDigitized` with the digitised
    time), or from the GPS UTC delta rounded to 15 minutes and accepted only within -12:00..+14:00;
  - (0, 0) and out-of-range coordinates mean no location;
  - a malformed date gives `None` and `exif_status = malformed`, never a failed ingest;
  - the broad-except ratchet does not grow.

**Ingest paths that must carry it** (one test per path): upload ingest, directory and bulk ingest,
dataset import (it reads the file), combine (copies `capture.*` verbatim from the source project).

**Job kind:** `capture_backfill`:
- queue `jobs`, exclusive group `capture_backfill`, scope `''`, auto-resume yes, 3 h / 3 h 15 m,
  max queue wait 24 h;
- a header-only read for images without `capture.parser`, through the storage seam `source` root;
- a missing source file is counted, not failed;
- it exists for upgraded 0.6.0 projects and for a future parser version.

**Wire (one serializer, `src/services/curation/wire.py`):**
- a `capture` object on the item detail model with the fields above, without location;
- one helper, `capture_visible_fields(policy)`, is the only place location could ever be added;
- a filter pair `taken_from` / `taken_to` (ISO dates, against `capture_taken_utc`, falling back to
  `capture_taken_local` with a documented floating-time note) on the existing crops list;
- `make contracts` in the same commit.

**Export:** label files and manifests never carry capture fields unless asked. The byte-copy path's
embedded GPS is J5.

### 5.2 (B) Doors the rebuild must keep open (tests to add)

| Door | Why it matters later | Test (owner lane) |
|---|---|---|
| `jobs.kind` without a CHECK | spatial and attribute kinds need no migration | already in 02a; keep |
| images index `index.knn = true` (a final setting) | a later `place_embedding` field is a mapping addition, not a reindex | `tests/opensearch/test_images_index_knn.py`: the bootstrap body sets it (2F) |
| project delete removes the whole `projects/<slug>/` prefix | a later `spatial/` area is purged with no delete-code change | `tests/clients/storage/test_delete_prefix_unknown_area.py`: an unlisted area under the prefix is removed (1E) |
| `op_scope_allows(pid)` for child tables | `scene_members`, `camera_poses`, `reconstruction_artifacts` reach the project through a parent | exists (02a 1.4) |
| the mapping-migration registry is additive and versioned | capture now, `attr_*` and `scene_ids` later | 2F owns it; `capture_fields.py` is its first additive user after the baseline |
| fixed region-box keys in one dataclass | `attr_*` keys are added to `RegionBox` in one place | exists (`tests/test_no_legacy_region_scalars.py` stays green) |
| explicit wire fields | no generic loop can serve a new sensitive field | a test that `capture.location` never appears in any serialized item or image (lands with 5.1) |
| queue consumer test covers profiles | a `gpu_spatial` queue later must have a consumer in its profile | exists (02b 7.3) |

### 5.3 (C) Reserved: attribute signals (after P-2/P-3 pass; Alembic revision `00NN_attribute_signals`)

Owner: the attribute lane after the tag (section 8, L-A1).

```sql
CREATE TABLE attribute_signals (               -- built-in, deployment-wide catalogue
  name        text PRIMARY KEY CHECK (name ~ '^[a-z][a-z0-9_]{1,31}$'),     -- 'colour', 'view'
  levels      text[] NOT NULL,                 -- subset of {'item','region_box'}
  value_kind  text NOT NULL CHECK (value_kind IN ('closed_vocab')),
  roles       text[] NOT NULL,                 -- allowed: subset of {'filter','stratum','assigner','soft','veto','must_link'}
  description text NOT NULL DEFAULT ''
);

CREATE TABLE attribute_signal_versions (
  id            uuid PRIMARY KEY DEFAULT uuidv7(),
  signal        text NOT NULL REFERENCES attribute_signals(name) ON DELETE RESTRICT,
  version       integer NOT NULL CHECK (version >= 1),
  method        text NOT NULL CHECK (method IN ('pe_zero_shot','pixel_lab','vlm_pack')),
  model_ref     text NULL,                     -- encoder name and revision for pe_zero_shot
  vocabulary    text[] NOT NULL,               -- e.g. the 11 basic colour terms; view bins
  params        jsonb NOT NULL,                -- prompts, abstain margin; pydantic-validated
  params_sha256 text NOT NULL,
  code_version  text NOT NULL,
  evidence_ref  text NOT NULL,                 -- the research note and benchmark key that passed the bar
  status        text NOT NULL CHECK (status IN ('candidate','active','retired')),
  created_at    timestamptz NOT NULL DEFAULT now(),
  created_by    text NOT NULL,
  UNIQUE (signal, version)
);
-- trigger attribute_signal_versions_immutable: params, vocabulary, method never change after insert

CREATE TABLE project_attribute_calibrations (  -- RLS (project_id)
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id         uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  signal_version_id  uuid NOT NULL REFERENCES attribute_signal_versions(id) ON DELETE RESTRICT,
  calibration        jsonb NOT NULL,           -- isotonic knots per label, abstain threshold
  metrics            jsonb NOT NULL,           -- dev-half accuracy, coverage, ECE
  sample_size        integer NOT NULL CHECK (sample_size > 0),
  job_id             uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  created_at         timestamptz NOT NULL DEFAULT now(),
  superseded_at      timestamptz NULL
);
CREATE UNIQUE INDEX uq_project_attribute_calibrations_live
  ON project_attribute_calibrations (project_id, signal_version_id) WHERE superseded_at IS NULL;
```

**Config kind:** `attribute_profile` (project-scoped; activation axis `attributes`; added to the
`config_documents.kind` and `config_activations.axis` CHECKs by the same revision). Body: the
enabled signals with their version, view prompts per coarse class (names, never ids), the enabled
roles (only those the signal row allows: fail closed).

**OpenSearch fields** (items and region-box elements, same keys):

| Field | Type |
|---|---|
| `attr_colour`, `attr_view` | `keyword` (a vocabulary value, or absent when abstaining) |
| `attr_colour_confidence`, `attr_view_confidence` | `float` (calibrated) |
| `attr_colour_signal`, `attr_view_signal` | `keyword` (`colour@2`) |
| `attr_colour_calibration`, `attr_view_calibration` | `keyword` (calibration uuid) |

**Jobs:** no new scoring kind: `colour` and `view` are `CropScorer`s in `item_scores` (the
`writes` ClassVar names exactly these fields). `attribute_calibrate` (queue `jobs`, exclusive
`attribute_calibrate`, scope = signal name). **Wire:** `attributes: {colour, view}`, each
`{value, confidence, signal}`. Filters `colour=`, `view=` on the crops list.

### 5.4 (C) Reserved: scenes and reconstructions (after an S1 GO and the owner's yes; revision `00NN_spatial_scenes`)

Owner: the spatial design doc that a GO triggers (section 8, L-SP1). All tables below are
project-scoped with RLS; child tables use `op_scope_allows` through their parent.

```sql
CREATE TABLE scenes (
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id         uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  display_name       text NOT NULL DEFAULT '',
  status             text NOT NULL CHECK (status IN ('candidate','confirmed','dismissed','archived')),
  current_version_id uuid NULL,                -- FK to scene_versions(id), DEFERRABLE
  revision           bigint NOT NULL DEFAULT 1,
  created_at         timestamptz NOT NULL DEFAULT now(),
  updated_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL
);

CREATE TABLE scene_versions (                  -- immutable once written
  id                  uuid PRIMARY KEY DEFAULT uuidv7(),
  scene_id            uuid NOT NULL REFERENCES scenes(id) ON DELETE CASCADE,
  project_id          uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  version             integer NOT NULL CHECK (version >= 1),
  pipeline_version    text NOT NULL,           -- descriptor@v + matcher@v + thresholds sha
  manifest_sha256     text NOT NULL,           -- sorted member image ids + imohash
  member_count        integer NOT NULL,
  verified_edge_count integer NOT NULL,
  readiness           text NOT NULL CHECK (readiness IN
                      ('high_potential','needs_more_views','spatial_memory_only','not_enough_evidence')),
  readiness_report    jsonb NOT NULL,          -- connectivity, verified share, baseline, preview registration
  job_id              uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  created_at          timestamptz NOT NULL DEFAULT now(),
  UNIQUE (scene_id, version)
);

CREATE TABLE scene_members (
  scene_version_id uuid NOT NULL REFERENCES scene_versions(id) ON DELETE CASCADE,
  image_id         text NOT NULL,              -- the images doc id in OpenSearch
  role             text NOT NULL CHECK (role IN ('member','representative','near_duplicate')),
  source           text NOT NULL CHECK (source IN ('machine','user')),
  PRIMARY KEY (scene_version_id, image_id)
);

CREATE TABLE scene_corrections (               -- user anchors; never lost on a re-run (P6)
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id         uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind               text NOT NULL CHECK (kind IN ('same_place','not_same_place','remove','merge')),
  image_ids          text[] NOT NULL CHECK (cardinality(image_ids) >= 1),
  scene_id           uuid NULL REFERENCES scenes(id) ON DELETE SET NULL,
  created_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,
  created_by_user_id uuid NULL
);

CREATE TABLE place_pair_evidence (
  project_id       uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  pipeline_version text NOT NULL,
  image_a          text NOT NULL,
  image_b          text NOT NULL,
  retrieval_rank   smallint NULL,
  retrieval_score  real NULL,
  inliers          integer NULL,
  inlier_ratio     real NULL,
  geometry         text NOT NULL CHECK (geometry IN ('essential','fundamental','homography','degenerate','none')),
  verdict          text NOT NULL CHECK (verdict IN ('verified','rejected','ambiguous')),
  reason           text NULL,
  job_id           uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  created_at       timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, pipeline_version, image_a, image_b),
  CHECK (image_a < image_b)
);

CREATE TABLE reconstruction_runs (
  id               uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id       uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  scene_version_id uuid NOT NULL REFERENCES scene_versions(id) ON DELETE RESTRICT,
  job_id           uuid NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE RESTRICT,   -- status lives on the job
  engine           text NOT NULL CHECK (engine IN ('colmap_global','colmap_incremental')),
  engine_version   text NOT NULL,              -- e.g. pycolmap-cuda12 4.2.1
  params           jsonb NOT NULL,
  artifact_prefix  text NOT NULL,              -- projects/<slug>/spatial/reconstructions/<id>/
  registered_count integer NULL,
  metrics          jsonb NULL,                 -- registered share, mean reprojection error, track length
  frame            text NOT NULL DEFAULT 'scene_local' CHECK (frame IN ('scene_local','enu')),
  geo_transform    jsonb NULL,                 -- Sim(3) scene_local -> ENU with its origin; null if not georegistered
  scale_known      boolean NOT NULL DEFAULT false,
  created_at       timestamptz NOT NULL DEFAULT now(),
  finished_at      timestamptz NULL
);

CREATE TABLE reconstruction_artifacts (
  id          uuid PRIMARY KEY DEFAULT uuidv7(),
  run_id      uuid NOT NULL REFERENCES reconstruction_runs(id) ON DELETE CASCADE,
  kind        text NOT NULL CHECK (kind IN ('colmap_sparse','pair_list','dense_points','mesh_glb',
                                            'splat_ply','splat_spz','preview','log')),
  storage_key text NOT NULL,
  bytes       bigint NOT NULL CHECK (bytes >= 0),
  sha256      text NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (run_id, kind, storage_key)
);

CREATE TABLE camera_poses (
  run_id                  uuid NOT NULL REFERENCES reconstruction_runs(id) ON DELETE CASCADE,
  image_id                text NOT NULL,
  registered              boolean NOT NULL,
  cam_from_world          real[] NULL CHECK (cam_from_world IS NULL OR cardinality(cam_from_world) = 12),  -- 3x4 row-major, COLMAP convention
  camera_model            text NULL,           -- COLMAP camera model name
  camera_params           real[] NULL,
  mean_reprojection_error real NULL,
  observations            integer NULL,
  unregistered_reason     text NULL,
  PRIMARY KEY (run_id, image_id)
);
```

Why these choices:
- The pose is a 3x4 matrix, not a quaternion, because quaternion component order differs between
  libraries.
- Poses live in Postgres: a viewer and the timeline read them per scene, and nothing searches them.
- The COLMAP model itself stays an artifact.
- Membership is projected to search as `scene_ids` on images (below) through the
  `search_projection_tasks` outbox (kind `scene_membership` added to its CHECK). This is the D-47
  pattern: one writer, a sync job, a verifier.

**OpenSearch:**
- images `scene_ids` (`keyword`, projection);
- `place_embedding` (`knn_vector`, dimension from its signal version) only if G-2 says a new
  vector;
- `capture.location` per D1a.

**Celery:**

| Kind | Queue | Exclusive group, scope | Limits |
|---|---|---|---|
| `place_descriptors` | `gpu_spatial` | `place_descriptors`, '' | 24 h |
| `place_pairs` (retrieve + verify) | `gpu_spatial` | `place_pairs`, '' | 24 h |
| `scene_grouping` | `jobs` (CPU) | `scene_grouping`, '' | 3 h |
| `scene_reconstruct` | `gpu_spatial` | `scene_reconstruct`, scene id | 24 h |
| `scene_localize` (S3) | `gpu_spatial` | `scene_localize`, scene id | 3 h |
| `scene_membership_sync` | `jobs` | `scene_membership_sync`, '' | 3 h |

Queue `gpu_spatial` consumer `spatial-worker`:
- its own image `docker/spatial/`: multi-stage, slim base, non-root user, `pycolmap-cuda12` pinned,
  Trivy before publish;
- profile `spatial`, prefork 1, `--max-tasks-per-child 1`, GPU `SPATIAL_GPU_ID` from `.env`;
- it imports only `src/tasks/runtime/` plus its handler, like the other GPU images;
- core and curation work with it absent.

**Storage (D-46):**
- area `projects/<slug>/spatial/` with `features/<pipeline_version>/` and `reconstructions/<run_id>/`;
- COLMAP's SQLite database is **job-local scratch**: built in the worker's temporary directory and
  never shared across jobs; results are copied in with `write_atomic`;
- with v0.8.0's S3 backend, `local_path(key)` must materialise to scratch; this plan notes it for
  that backend's design.

**Config kind:** `spatial_profile` (axis `spatial`):
- retrieval arm and K; matcher; inlier and degeneracy thresholds (calibrated, with a calibration
  id); minimum scene size;
- a per-run budget (maximum pairs, maximum images per reconstruction);
- the opt-in for automatic reconstruction (off by default).

**API (none exists; written without a leading slash):**
- `curation/projects/{project}/scenes` (GET list by readiness);
- `.../scenes/discover` (POST, 202 + job);
- `.../scenes/{scene_id}` (GET), `.../scenes/{scene_id}/members` (GET), `.../scenes/{scene_id}/evidence` (GET, the overlap graph);
- `.../scenes/{scene_id}/corrections` (POST, `expected_revision`);
- `.../scenes/{scene_id}/reconstructions` (POST 202, GET);
- `.../reconstructions/{run_id}` (GET), `.../reconstructions/{run_id}/artifacts/{artifact_id}` (GET, streamed through the storage seam);
- `.../timeline` (GET, `date_histogram` plus coarse `geotile_grid`; only with D1a and J13).

---

## 6. Prototype protocol

### 6.1 Code location, venv, pins

- **Code: tracked, in `scripts/research/spatial/`**, with tests in `tests/research/`.
  - Why tracked and not gitignored: the research rules require a reproduce command from a clean
    checkout. The repo already tracks research scripts with tests (`scripts/research/`,
    `tests/research/`). The naming-leak scan and the layering ratchets then apply to them.
  - Paths are arguments, never constants, so no set P path can enter the tree.
  - Outputs go to `artifacts_local/` (gitignored).
  - Nothing under `src/` imports them.
- **Pins:** `scripts/research/spatial/requirements.txt` with **exact** `==` pins. This is a
  deliberate change from the unpinned `scripts/research/requirements.txt`, under D-44.
  - Candidates as of 2026-11-04: `pycolmap-cuda12==4.2.1`, plus `torch` and `torchvision` matching
    the CUDA 12 wheel. The exact versions of those, `faiss-cpu`, `scipy`, `scikit-image`,
    `statsmodels`, `networkx` and `pyexiftool` are resolved and frozen at P-0.
  - The MegaLoc `torch.hub` commit is pinned by SHA.
  - `pip freeze` is saved with every run.
  - No SuperPoint, and no weights without a stated licence, except the measurement-only arms the
    owner allows (J7).
- **Venv:** a venv outside the repository (the research scratch directory), Python 3.12. Install
  `requirements-test.txt`, then the spatial file. No Docker build, no host-wide install. ExifTool
  comes from the OS package for the P-1 oracle only.

### 6.2 Prototypes

Each prototype's bar is committed in a registration note before any score is read. Registration
note: `docs/research/spatial_attributes_registration.md`. Results note:
`docs/research/spatial_attributes_results.md`.

| ID | Question | Data | Arms | Metrics | Bar (registered at P-0) | Compute |
|---|---|---|---|---|---|---|
| P-0 | registration, pins, fetch checks | - | - | - | committed before any score | docs |
| P-1 capture | Is Pillow's capture parse correct and complete, and how much metadata do real libraries carry? | ExifTool's sample images and the public sets in 6.3 (census); YFCC100M metadata and its Exif pack (field distribution, metadata only); set P (owner-run, counts only) | Pillow parser vs ExifTool oracle; ExifRead for formats Pillow cannot open | per-field agreement with ExifTool (time to the second, offset, lat/lon to 1e-6 deg, focal); parse failures; µs per header; coverage shares (time, offset, GPS, XMP-only date) | agreement >= 0.999 on time and offset where ExifTool reads them; 0 ingest failures on malformed files; p95 header parse <= 1 ms; XMP fallback (D1c) only if XMP-only dates exceed a registered share | CPU, minutes |
| P-2 view | Can a zero-shot view tag from stored PE vectors reach the bar? | Pascal3D+ (car, bus, motorbike, bicycle; azimuth to 5 bins: front, front three-quarter, side, rear three-quarter, rear); CompCars local cross-check | PE zero-shot prompts; PE linear probe on Pascal3D+ train; VLM pack; Orient Anything only if J7 | per-bin accuracy with Wilson 95 % intervals; macro accuracy; coverage at the abstain margin; ECE after isotonic calibration | >= 0.8 per bin, >= 60 crops per bin, coverage >= 0.5 (the former #400 plan) | GPU about 0.5 h to encode crops; CPU after |
| P-3 colour | Which colour arm is accurate and stable? | PACO-LVIS colour attributes on COCO images (licence of annotations checked first, J10); illumination jitter of the same crops (white balance, exposure) as a flip test; near-duplicate pairs once #416 has a threshold | PE zero-shot; pixel Lab + CIEDE2000 to 11 terms (mask when available); VLM pack | top-1 accuracy over 11 terms (multi-valued ground truth: any match); flip rate under jitter and on near-duplicate pairs; coverage | accuracy registered at P-0 from the dev half; flip rate <= 10 % (the former #400 plan) | GPU about 0.5 h; CPU |
| P-4 retrieval (S1a) | Do stored vectors find same-place photos? | IMC test scenes + Cambridge (test), IMC train (dev); GLDv2 and COCO distractors with the contamination filter | PE (product preprocessing), MobileCLIP, MegaLoc; SALAD or AnyLoc only if J7 | recall@K of co-visible and same-place pairs (K 10, 20, 50); precision of top-K; encode cost | G-1 recall@20 >= 0.90; G-2 reuse PE if within 0.05 | GPU about 2-3 h |
| P-5 verification (S1b) | Do retrieved pairs share real geometry? | top-K pairs of the best arm | pycolmap SIFT + `SIFT_LIGHTGLUE`; `ALIKED_N16ROT` + `ALIKED_LIGHTGLUE`; SIFT + ratio test; Doppelgangers only if G-3 fails | edge precision on look-alike and cross-scene pairs; edge recall on co-visible pairs; per-pair cost | G-3 zero cross-scene verified edges and precision >= 0.99; G-4 recall >= 0.80 | GPU about 1 h |
| P-6 grouping (S1c) | Do verified edges give pure groups? | P-5 edges | components; 2-core | BCubed P/R vs scene id; cross-scene merges (count); fragmentation | G-5 P >= 0.98, R >= 0.80, 0 merges | CPU minutes |
| P-7 reconstruction (S1d) | Does a discovered group reconstruct like a curated one? | largest discovered component per IMC test scene | `pycolmap.global_mapping`; `incremental_mapping` | registered share vs the reference model; mAA at the IMC thresholds; wall time; peak RSS | G-6 >= 0.90 registered on >= 5 of 7 scenes; mAA within 0.05 of the curated run | CPU 2-6 h |
| P-8 priors and cross-time (S1e + R4) | Do GPS and time priors help, and does it hold across time? | MSLS (GPS, capture time, compass angle; J10); Aachen day/night; RobotCar seasons; a YFCC100M geotagged sample around public landmarks | vision-only vs GPS-radius pre-filter vs both | recall@K and edge recall per condition; candidate count saved | reported, not gated | GPU about 0.5 h |
| P-9 set P (owner-run) | Do P-1 and P-4..P-6 hold on a real personal library? | set P, frozen sample | best arms only | counts and rates only | reported; no location value ever recorded (6.5) | owner's window |

Ground-truth rules carried from the former #348 plan:
- co-visibility: at least `T_cov` shared 3D points in the reference COLMAP model, suggested 100,
  registered at P-0;
- same-place but not co-visible pairs are reported apart;
- the GLDv2 contamination filter (drop landmark categories matching IMC or Cambridge scenes, then a
  person checks 50 neighbours per scene);
- IMC train is dev, IMC test plus Cambridge are test, read once;
- G-6 is read only on discovered groups, never on reference image lists.

Verdicts per stage: PASS, FAIL or INCONCLUSIVE, never rounded.

### 6.3 Public datasets

| Set | State | Licence note | Used by |
|---|---|---|---|
| IMC PhotoTourism, Cambridge, GLDv2 subset, COCO val2017, Aachen v1.1, RobotCar Seasons v2 | fetched and verified ([spatial_datasets.md](../../research/spatial_datasets.md)); re-run `fetch_spatial_sets.py verify --test-archives` before use | per that note; several non-commercial | P-4..P-8 |
| Mapillary Street-Level Sequences ([paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Warburg_Mapillary_Street-Level_Sequences_A_Dataset_for_Lifelong_Place_Recognition_CVPR_2020_paper.pdf), [repo](https://github.com/mapillary/mapillary_sls)) | **not fetched**; needs a sign-up | CC BY-NC-SA plus Mapillary terms, **including a confidentiality obligation**: aggregates only, nothing derived committed | P-8 (GPS, time, heading ground truth) |
| YFCC100M metadata and the Exif expansion pack ([paper](https://arxiv.org/abs/1503.01817), [expansion packs](https://multimediacommons.wordpress.com/yfcc100m-expansion-packs/)) | **not fetched** | metadata per the dataset terms; images under their own CC licences | P-1 (real-world field distribution), P-8 (geotagged sample) |
| Pascal3D+ | **not fetched** | PASCAL VOC and ImageNet image terms; research | P-2 |
| PACO-LVIS ([paper](https://arxiv.org/abs/2301.01795)) | **not fetched** | code MIT; **annotation licence to verify**; COCO images | P-3 |
| CompCars | not fetched | research licence; local only | P-2 cross-check |
| MegaScenes (Tung et al., ECCV 2024) | not fetched; candidate | Wikimedia Commons licences per image; verify | a second discovery set if IMC is too small (J10) |

The catalogue in `spatial_datasets.md` gains these rows when they are fetched (section 11).

### 6.4 Repetition, GPU etiquette, stop rules

- **Repetition:**
  - three seeds wherever sampling happens (distractors, visual-check samples, calibration splits);
  - exact kNN and fixed COLMAP seeds otherwise;
  - P-7 repeated once to measure spread;
  - timing rows three times with the load average beside each;
  - a gain counts only if it exceeds twice the run-to-run spread, with a 1,000-resample bootstrap
    interval that excludes 0 (`scipy.stats.bootstrap`).
- **GPU etiquette:**
  - GPUs 0 and 2 only; never GPU 1;
  - one GPU job at a time on the host, serialized with `flock` on the shared lock file (the
    `docs/research/CLAUDE.md` rule);
  - message the peer sessions and wait for their reply before a GPU window;
  - record `uptime` and `nvidia-smi` at the start and end;
  - stop at once when a peer asks;
  - no wait loops (no `pgrep -f`, no sleep-polling); long runs go in the background with a
    completion notice;
  - any compose stack runs as `-p op-rs-spatial`, never the default project (none is needed: the
    prototypes are venv scripts).
- **Stop rules:**
  - P-1: if Pillow agreement is under the bar, try ExifRead; if both fail, stop D1 and report.
  - P-2 or P-3: an arm under its bar does not ship; if every arm fails, the signal stays off and
    the outcome note says why.
  - P-4 G-1 fail, or P-5 G-3/G-4 fail, or P-6 G-5 fail: NO-GO on discovery; #348 stays parked;
    alternatives (GLUEMAP, hloc) only for a failure specific to them.
  - P-7 fail with discovery passing: "discovery only" stays a candidate (J12).
  - Any prototype that runs over its registered compute by 2x: stop and report.

### 6.5 Recording results

- **Public numbers:**
  - `docs/research/spatial_attributes_results.md` (Setup, Method, Results, Findings, Open issues,
    Reproduce);
  - backing JSON in `docs/benchmarks/spatial_attributes/` with `SHA256SUMS`; additions only,
    never pruned;
  - git sha, pin hash, `pip freeze`, GPU id, driver, load average and wall time on every row;
  - scorecards through `scripts/bench/scorecard.py` with `synthetic: false`.
- **Non-commercial sets** (Aachen, RobotCar, MSLS, IMC Flickr photos, 7-Scenes): aggregate tables
  only; no images or derived images in the repo. MSLS's confidentiality term means only aggregate
  metrics.
- **Visual checks:**
  - a person reviews match drawings, merges and rejected co-visible pairs, with the local VLM as a
    ranking aid at most;
  - tourist and street photos contain faces and plates, so they never go to a hosted model;
  - the hosted judge may see only public, face-free and plate-free contact sheets under owner
    decision O3, and never anything from set P.
- **Set P, and set P location above all, never leaves the machine:**
  - no coordinates, place names, maps, reverse-geocoded labels, per-image times or file names in
    any commit, gist, issue, comment, log shipped off the host, or prompt to any hosted model;
  - committed numbers are coverage shares and rates only, for example "share of set P images with
    a GPS fix", never a value;
  - capture-date distributions are published only at a coarseness that cannot reveal travel (year
    counts at most, owner's call);
  - raw outputs stay in `artifacts_local/` with owner-only permissions;
  - no reverse geocoding or timezone lookup is run on set P coordinates;
  - the naming scan does not catch numbers or coordinates, so the committer checks every set P row
    by hand.

---

## 7. Production design

### 7.1 Capture (v0.7.0)

1. The decode helper opens the header, as it already does for orientation, and reads IFD0, the Exif
   IFD and the GPS IFD into plain dicts. `parse_capture` turns them into `CaptureMetadata`.
2. The ingest writer writes the `capture.*` fields on the images doc and the two copies on each
   item, in the same bulk write ingest already makes (no extra round trip). Location is dropped by
   the writer unless D1a is approved and the project allows it.
3. `capture_backfill` runs on upgraded projects and on a parser version bump. It reads headers
   only, through the `source` root, under its exclusive job, with a resumable checkpoint (the last
   image id).
4. Readers in v0.7.0: the item detail `capture` object, and the `taken_from` / `taken_to` crop
   filter.
5. Cost: one dict walk per image on a header already parsed. P-1 measures it before the W5a
   baseline, so the baseline includes it.

### 7.2 Attributes (after the tag, after P-2/P-3)

- `colour` and `view` scorers in `item_scores`:
  - view: one text encode per prompt per project (cached by `params_sha256`), then a matrix product
    over stored PE crop vectors (CPU);
  - colour: PE zero-shot the same way, or the pixel arm on the crop decoded at ingest. The pixel arm
    never decodes a second time, so a backfill of that arm is a measured, separate cost.
- Calibration (`attribute_calibrate`):
  - isotonic regression on a seeded, person-labelled or dataset-labelled dev sample per project and
    signal version;
  - near-duplicates never straddle the split;
  - a version with no live calibration writes no value (fail closed).
- How attributes feed search and clustering:

| Use | colour | view | capture time | location (later) |
|---|---|---|---|---|
| search filter (crops list) | yes | yes | yes (v0.7.0) | only via timeline routes, policy-gated |
| clustering, class level | `stratum` only | `assigner` and `stratum` | none | none |
| clustering, identity level (#346 RQ-6, owner-run) | soft `veto` only with weak embedding similarity | never a veto | burst `must_link` only with high similarity (bursts within one camera make and model) | never for crops |
| scene discovery | none | none | session candidates, never proof | radius candidates, never proof |
| export, training | never | never | never unless asked | never unless asked and allowed |

- The rules from the former #400 plan stand, as tests in `src/domain/attribute_edges.py` when the
  first signal passes:
  - no attribute decides a merge alone;
  - roles and levels come from the registered signal row (fail closed);
  - class compatibility is by name;
  - one vote per near-duplicate group;
  - provenance `attribute:<name>@<version>`; never sets `*_validated`;
  - deterministic order.
- The edge builder has no consumer until RQ-5 machinery exists (#346). It is written with its first
  consumer, never before.

### 7.3 Scenes (after an S1 GO)

1. **Candidates.**
   - `place_descriptors` encodes images (or reuses PE per G-2);
   - `place_pairs` retrieves top-K inside the project only (the project guard and RLS bound it),
     optionally pre-filtered by time sessions and the location radius, then verifies with pycolmap;
   - every pair writes `place_pair_evidence`, verified or rejected, with a reason;
   - incremental: a new image retrieves and verifies against representatives only, with a budget
     cap from `spatial_profile`.
2. **Groups.**
   - `scene_grouping` builds 2-core components over verified edges and applies
     `scene_corrections` as hard anchors (a user's "not the same place" is a cannot-link that no
     machine edge overrides);
   - it writes a new `scene_versions` row when membership changes;
   - ids carry over by membership overlap; readiness is a report.
3. **Reconstruction**, opt-in per scene by default:
   - `scene_reconstruct` freezes the version's manifest, runs `pycolmap.global_mapping` (incremental
     as the fallback arm per P-7) in job-local scratch, then writes `reconstruction_runs`,
     `camera_poses` and `reconstruction_artifacts` through the storage seam;
   - optional geo-registration (`model_aligner`) only when the location policy stores location;
     otherwise `frame = scene_local`, `scale_known = false`.
4. **Timeline levels** (the spatial note, section 10):
   - level 1, the geo-photo timeline from capture metadata (D6);
   - level 2, posed photos in a reference scene (S3, `scene_localize` with hloc);
   - level 3, multi-year reconstructions only when each period has the coverage.
   - Unlocalized photos stay on the timeline with a lower-confidence marker. Inferred geometry is
     labelled as inferred.
5. **Decode once:** feature extraction runs at its own resolution, so it is a second read of the
   file. The spatial design doc records it as a reasoned exception for the owner to approve, or
   reuses the resident frame (the owner's one-decode rule).

---

## 8. Landing lanes

| Lane | Wave | Branch | Owns | Preconditions | Tests |
|---|---|---|---|---|---|
| L-C1 capture metadata | **W4**, order 2 (cut after `b4/bulk-occ` merged, since 4A rewrites the ingest data plane) | `b4/capture-metadata` | `src/domain/capture_metadata.py` (new); the header read in `src/services/curation/image_decode.py`; the capture part of the ingest writer; `src/clients/curation_opensearch/capture_fields.py` (new) and its mapping-migration registration; the `capture` wire block and the `taken_from`/`taken_to` filter; `src/config/job_kinds/capture.py` and its task handler; `tests/curation/test_capture_*.py`, `tests/domain/test_capture_metadata.py`; docs (`docs/CURATION.md`, `curation_api_contract.md`), `make contracts`, `CHANGELOG.md` | B2 merged; J1, J2, J4 answered; P-1 bars passed (P-1 is CPU, can run before W4) | red/green per rule in 5.1; each ingest path carries `capture`; the location canary never appears in any serialized response or export; malformed EXIF never fails ingest |
| L-C0 schema bit (only if J2 = opt-in now) | W2.1 lane 1A + W2.2 lane 2C/2E | existing | `project_settings.location_policy` in `0001`; the settings model and route; the upgrade writes `drop` | J2 | `tests/db/test_schema_matches_plan.py`; settings OCC test |
| L-X1 export orientation fix | **W5b**, track `decode` (5b-D1) | existing | `src/services/curation/export_support.py::_copy_or_resize_one` through the oriented decode helper | 5b-D1 | red: an orientation-6 fixture exported with `resize_mode` set lands boxes on the object; control: tag 1 byte-identical |
| L-X2 export metadata | W4, inside L-C1 (or the privacy lane, coordinator's call) | `b4/capture-metadata` | per J5: lossless GPS strip on exported copies, or a documented statement | J5 | an exported copy of a GPS fixture has no GPS IFD (or the doc says it keeps it) |
| R-1 research P-0..P-3 | after the tag, CPU slots plus one short GPU slot | `research/spatial-attributes` | `scripts/research/spatial/**`, `tests/research/test_spatial_*.py`, the registration and results notes, `docs/benchmarks/spatial_attributes/` | MQ-9 (research after the tag); J7, J10 | registration before results (`git log`); scorecards |
| R-2 research P-4..P-8 (S1) | after the tag, a research window of about 4 GPU h and 10 CPU h | same branch | same | J9 | same |
| L-A1 attribute signals | after R-1 passes for a signal | `feat/attribute-signals` | revision `00NN_attribute_signals`; scorers; `attr_*` mapping; wire and filters; `attribute_profile` kind | R-1 PASS for that signal | the former #400 plan's 8.3 tests for the parts that land |
| L-T1 geo-photo timeline | v0.8.0 | `feat/capture-timeline` | location fields per D1a; `capture_location_purge` job; timeline route | J2, J13 | location purge leaves no trace (counts-only audit); policy change to `drop` triggers the purge |
| L-SP1 scenes | after an S1 GO and the owner's yes; its own design doc first | per that doc | section 5.4 | GO; J12, J14 | per that doc |

No lane runs a full suite. The coordinator merges with `--no-ff` and runs one sweep per wave.

---

## 9. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| **Location privacy** | GPS reveals homes, routines and the movements of the people photographed | Neutral rules, all fail closed: (1) no coordinates stored in v0.7.0 (recommended); (2) if stored later, only under a per-project policy whose default is `drop`, and an upgraded project gets `drop`; (3) one visibility helper on every read path, with a test that enumerates them (the Immich lesson); (4) location never copied to items, never in labels, manifests or training files, never sent to any VLM or hosted model; (5) switching to `drop` purges the stored values (a job with a counts-only audit); project delete removes them; (6) derived location (reverse geocode, time zone from coordinates, scene geo-transform) follows the same policy as the coordinates; (7) the API has no authentication, so "served" means served to anyone who can reach it; SECURITY.md says so |
| **Location inside original files** | exports by byte copy and stored uploads keep embedded GPS whatever we store | J5: strip GPS losslessly on export copies (ExifTool), or state it plainly in SECURITY.md and the export docs. Stored originals stay byte-identical (dedup and provenance) and are as private as the deployment |
| Set P location in public text | an irreversible leak | 6.5 rules; the committer checks set P rows by hand; nothing from set P to hosted models |
| Wrong capture times (no offset, camera clock wrong, edited files) | timeline and bursts mislead | store the source, precision and offset source; never treat file mtime or IFD0 `DateTime` as capture time; no cross-camera comparison without offsets |
| Malformed EXIF fails ingest | lost images | catch the named exceptions only; `exif_status = malformed`; test with fuzzed headers |
| Colour flips with lighting | false vetoes, wrong filters | flip-rate bar; class-level colour is never a veto; abstain under the margin |
| View symmetry and occlusion | wrong tags | per-bin bar and coverage floor; never a veto |
| Repeated structure | merges two places | G-3 zero cross-scene edges; 2-core; Doppelgangers arm; visual review of every merge |
| Licences (SALAD weights, SuperPoint, Orient Anything, colour-name table, PACO annotations, GeoCalib) | an unshippable default | measurement-only until a licence is confirmed; ATTRIBUTION before anything ships |
| Non-commercial and confidential datasets | a terms breach | aggregates only; MSLS confidentiality; no derived images committed |
| GPU contention | slow, noisy rows; peers blocked | GPUs 0/2, one job, `flock`, peer messages, load recorded |
| Scope creep into a viewer before discovery works | months of UI on an unproven base | the go/no-go gates; no route or UI in research |
| Storage growth (features, reconstructions, place vectors) | disk pressure | artifacts under the project prefix with sizes recorded; G-7 cost rows size the job design; place vectors only if G-2 says so |
| Decode twice for features | violates the one-decode rule silently | recorded exception for the owner, or reuse of the resident frame (7.3) |

---

## 10. Owner questions

| J | Question | Recommendation | Trade-off |
|---|---|---|---|
| J1 | Should capture metadata (time, offset, camera, lens, focal, `has_location`) land in the v0.7.0 rebuild, as lane L-C1 in W4, with the `taken_from`/`taken_to` filter as its reader? | **Yes.** It is the only signal waiting can lose, and it is cheap and standard | A small new surface in a program that otherwise adds no features |
| J2 | Store GPS coordinates in v0.7.0? | **No.** Keep `has_location` only. Add coordinates with the timeline or scenes (L-T1), behind `location_policy` (default `drop`; values `drop`, `store_coarse`, `store`) | A later backfill needs the source files still present |
| J3 | Camera body serial? | **Never stored raw.** A keyed hash only if bursts need it, reusing #401's secret design. Make plus model until then | Two bodies of one model share a burst scope |
| J4 | Run `capture_backfill` on upgraded 0.6.0 projects? | **Yes**, header-only, through the `source` root; a missing file is counted | Source-root I/O once per image |
| J5 | Exported originals carry embedded GPS today (byte copy). Strip it? | **Strip GPS losslessly on export copies** with ExifTool in the jobs image, unless the export request asks to keep it. Otherwise document it | A new system dependency in the image, against a known leak |
| J6 | Colour and view in v0.7.0 or after the tag? | **After the tag**, each after its P-2/P-3 bar | No attribute filters at the release |
| J7 | Measurement-only arms with unclear licences (SALAD weights, AnyLoc ViT-G, Orient Anything, the colour-name table, GeoCalib)? | Allow them locally for measurement, never as a product default | Without them a MegaLoc loss cannot be read as "VPR does not help" |
| J8 | Matcher stack? | **pycolmap only** (SIFT or ALIKED with COLMAP's LightGlue). Kornia only if throughput is under about 20 pairs/s | One dependency vs a second view of matcher quality |
| J9 | When does S1 (P-4..P-8) run? | After the tag, as a rider in a clustering research window (it shares the "encode once" step with #346's E-12) | About 4 GPU hours and 2 person hours from other research |
| J10 | Fetch MSLS (sign-up, confidentiality), the YFCC100M metadata and Exif pack, Pascal3D+, PACO-LVIS (after its annotation licence is checked), optionally MegaScenes? | **Yes**, local only, aggregates only | Disk and download time; terms to accept |
| J11 | Which prototypes run on set P, and do you confirm the set P location rule of 6.5? | P-1 census and P-9 (discovery arms) owner-run; confirm the rule | Owner time; no public numbers from them |
| J12 | Dense, splat and viewer (S2), OpenMVS? | Decide after a GO. COLMAP dense plus gsplat first; OpenMVS only for mesh quality, with its AGPL terms reviewed | Deferral vs early visual impact |
| J13 | A geo-photo timeline (level 1) and reverse geocoding (GeoNames in Postgres, Immich's approach) in v0.8.0? | Timeline yes, with J2's policy. Reverse geocoding only if the timeline needs place names; it is derived location under the same policy | GeoNames data size and refresh |
| J14 | Where does a scene product live? | Curation, project-scoped (guard, RLS, delete semantics), not the core API | Core-only users need a project |
| J15 | Carried from the former #400 plan: run E-8b/E-8c on the research harness without waiting for RQ-5 (its J1), and file "one vote per near-duplicate group" as a separate loop item (its J2)? | Yes to both | As recorded there |

---

## 11. Required changes to other plan files (for the integration agent)

| File | Change | Why |
|---|---|---|
| [program/00_master_plan.md](00_master_plan.md) W4 table | add lane "Capture metadata at ingest" `b4/capture-metadata`, order 2 (after 4A merged), owns per section 8 L-C1; merge order 4A, 4D, privacy lane, capture lane, 4B | J1 |
| [program/05_b4_stateless_api.md](05_b4_stateless_api.md) | a track for L-C1 with the red tests of 5.1; the privacy lane's SECURITY.md text gains the location rules of section 9 (and J5's export statement if not stripped) | J1, J5 |
| [program/02b_jobs_locking_events.md](02b_jobs_locking_events.md) section 2.1 | kind row `capture_backfill`: project yes, queue `jobs`, `capture_backfill`, '', auto-resume yes, 3 h / 3 h 15 m, 24 h | J4 |
| [program/02a_postgres_schema.md](02a_postgres_schema.md) section 4 | only if J2 = opt-in now: `project_settings.location_policy` (5.1); otherwise nothing. Add a note that 5.3 and 5.4 here hold the reserved DDL for later revisions | J2 |
| [program/03a_stage1_foundation.md](03a_stage1_foundation.md) lane 1E | the test `test_delete_prefix_unknown_area.py` (5.2) | door B |
| [program/03b_stage2_control_data.md](03b_stage2_control_data.md) lanes 2C, 2E, 2F | 2C/2E only if J2 = opt-in now (settings field; the upgrade writes `drop`). 2F: the images body keeps `index.knn = true`, with `test_images_index_knn.py`; the mapping-migration registry is additive and versioned (its first post-baseline user is `capture_fields.py`) | doors B |
| [program/06b_b5_decode_once.md](06b_b5_decode_once.md) 5b-D1 | add `src/services/curation/export_support.py::_copy_or_resize_one` to "Callers to adapt" (no `exif_transpose` on the resize and crop path: wrong pixels for tags 2-8); the move of `decode_image` to `src/utils/image_decode.py` carries the capture header read and `DecodedImage.capture` | section 2.1 finding; L-X1 |
| [program/00b_traceability.md](reference/traceability.md) | map #348 and #400 to this plan (research after the tag; L-C1 in W4 if J1 = yes); record the export orientation bug under 5b-D1 and the export GPS item under J5 | traceability |
| [program/09_github_actions.md](reference/github_actions.md) | issue comments for #348 and #400 pointing at this file (plan only; GitHub is not changed by this pass) | the issues still point at the old gists |
| [program/10_standards_and_libraries.md](reference/standards_and_libraries.md) section 3 | register rows: capture metadata (Pillow `getexif`/`get_ifd`; ExifTool as the research oracle and the export stripper); place retrieval (MegaLoc via `torch.hub`, measured against PE); verification and SfM (pycolmap); colour (scikit-image Lab and CIEDE2000, Pillow quantize); calibration (scikit-learn `IsotonicRegression`) | library register |
| [09b_research_clustering.md](09b_research_clustering.md) | RQ-6 row and E-8 link here (E-8 split a-d stands; the view ground truth becomes Pascal3D+ first, person labels second); attribute roles per the table in 7.2 | #400 moved here |
| [10_v080_platform_breadth.md](10_v080_platform_breadth.md) section 6, the #348 trigger | replace "the clustering research plan reports its place-recall results" with: the v0.7.0 tag exists, a research window has about 4 GPU hours on GPU 0 or 2, and the owner says go (J9); S1 is P-4..P-8 of this plan. Add L-T1 (timeline) as a v0.8.0 candidate under J2/J13 | stale trigger (the former #348 plan, C5) |
| [09b_research_clustering.md](09b_research_clustering.md) | where it refers to the #348 and #400 plans, point at this file | the plans are deleted |
| [../research/README.md](../../research/README.md) | the spatial note row: drop "milestone v0.9.0" (#348 has no milestone); add rows for the registration and results notes when they exist | stale text |
| [../research/spatial_datasets.md](../../research/spatial_datasets.md) | add MSLS, YFCC100M metadata and Exif pack, Pascal3D+, PACO-LVIS, MegaScenes as rows once fetched (J10), with licences | 6.3 |
| [README.md](../README.md) (design index) | done in this commit: the two per-issue rows are replaced by this plan's row | links |

---

*Plan only. Sources are cited inline. Library and dataset terms were read on 2026-11-04 and are
re-checked before anything ships.*
