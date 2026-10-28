# OpenProcessor Spatial Memory & Automatic 3D Reconstruction

**Consolidated research, GitHub comparison, technical architecture, and implementation plan**  
**Research date:** October 28, 2026  
**Status:** Research and proposed design — not a claim that the 3D features are implemented in OpenProcessor.

> **The product concept:** An ordinary photo library should *discover* groups of images depicting the same physical place, determine which groups can be reconstructed in 3D, optionally generate a high-quality 3D scene, and display photographs from different years in their estimated spatial positions. Users should not have to export, sort, and manually submit every image set to a separate photogrammetry application.

## Contents

1. [Executive summary and decisions](#1-executive-summary-and-decisions)
2. [Inspiration: God's Eye View and the Geospatial Memory Palace](#2-inspiration-gods-eye-view-and-the-geospatial-memory-palace)
3. [Important technical distinctions](#3-important-technical-distinctions)
4. [Finished applications: maturity, stack, deployment, quality](#4-finished-applications-maturity-stack-deployment-quality)
5. [Geometry/reconstruction engines](#5-geometryreconstruction-engines)
6. [Gaussian splat trainers, formats, and viewers](#6-gaussian-splat-trainers-formats-and-viewers)
7. [How the output quality actually compares](#7-how-the-output-quality-actually-compares)
8. [What OpenProcessor already has — and the missing product layer](#8-what-openprocessor-already-has--and-the-missing-product-layer)
9. [Proposed automatic scene-discovery algorithm](#9-proposed-automatic-scene-discovery-algorithm)
10. [Historical photographs and a four-dimensional memory timeline](#10-historical-photographs-and-a-four-dimensional-memory-timeline)
11. [Deployable architecture, storage, API, and workers](#11-deployable-architecture-storage-api-and-workers)
12. [Model and engine selection](#12-model-and-engine-selection)
13. [Quality gates, benchmarking, and failure modes](#13-quality-gates-benchmarking-and-failure-modes)
14. [Mac/Linux hands-on evaluation](#14-maclinux-hands-on-evaluation)
15. [Licensing, privacy, and dependencies](#15-licensing-privacy-and-dependencies)
16. [Implementation phases and estimated effort](#16-implementation-phases-and-estimated-effort)
17. [Complete research directory](#17-complete-research-directory)
18. [Open questions and recommended first experiment](#18-open-questions-and-recommended-first-experiment)

---

## 1. Executive summary and decisions

**Recommended default stack:**

| Layer | Preferred starting point | Why | Alternative / experiment |
|---|---|---|---|
| Existing image processing | **OpenProcessor**: FastAPI + Triton + OpenSearch | Preserve image ingestion, detection, embeddings, search, clustering | — |
| Discovery of candidate same-place photos | Existing MobileCLIP + **SALAD** place descriptors + GPS/EXIF | Semantic and geographic filters reduce the search space | MegaLoc / DINO visual place retrieval |
| Verify real image overlap | **SIFT or ALIKED + LightGlue + robust geometric estimation** | Converts semantic neighbors into evidence of common visible geometry | COLMAP 4.2 LoMa learned features/matching |
| 3D camera geometry | **COLMAP + pycolmap** | Established baseline; compatible tools, camera poses, sparse cloud | OpenSfM 1.0 for geography; GLUEMAP/VGGT for hard cases |
| Photorealistic world | **gsplat** on NVIDIA GPU | Scriptable CUDA trainer, strong research/engineering ecosystem | Brush for portability; OpenSplat for cross-platform/CPU |
| Physical surfaces | COLMAP dense pipeline; evaluate **OpenMVS** if license fits | Dense clouds, textured meshes and geometry | AliceVision, ODX/MicMac for survey-style work |
| Web splat display | **PlayCanvas / SuperSplat ecosystem** | Polished real-time splat viewing, runtime formats | Spark/Three.js |
| Geographical presentation | **CesiumJS** | Map-scale geospatial coordinates, terrain, image locations | MapLibre for 2D-first maps |
| Temporal memory | New OpenProcessor scene/camera/time model | Register old images to stable reference scene; scrub chronology | Chronoscope concepts for temporal provenance |
| Deployment | Optional GPU worker(s) in Docker Compose | Core image search works without expensive 3D services | EKS/Kubernetes later |

**Bottom line:** Do not rewrite COLMAP or a Gaussian renderer. Build the missing *automatic photo-library-to-reconstruction-dataset* capability, benchmark a conventional geometry pipeline first, and make the 3D/time viewer optional and discoverable in the existing gallery.

**What we did not find:** A mature, broadly adopted, self-hosted photo app with all of the following as one unattended workflow: arbitrary library ingest → automatic spatial-overlap detection → reconstruction readiness → durable 3D jobs → geometry/splats → per-year image alignment → interactive spatial memory timeline. Individual portions exist; the integration is the opportunity. This is a qualified assessment, **not proof that no such project exists anywhere**.

**Evidence standard:** Product statements below are from project README/docs/release descriptions, not hands-on benchmarks or independent stability testing. Quality and development time ratings are engineering judgments. Check release notes and licenses again before shipping a commercial product.

## 2. Inspiration: God's Eye View and the Geospatial Memory Palace

**Creator:** [Bilawal Sidhu](https://www.youtube.com/@bilawalsidhu).  
**Open-source globe:** [bilawalsidhu/gods-eye-view](https://github.com/bilawalsidhu/gods-eye-view).  
**Concept:** Geospatial Memory Palace: photos and video memories placed in 3D space at their locations and captured times, sometimes comparing the *same physical place* across seasons/years.

Relevant videos and research leads:

- **"Turning 2D Photos Into 3D Holograms. The Spatial Memory Palace Is Coming To Life!"** — August 20, 2026; [creator channel](https://www.youtube.com/@bilawalsidhu) and [upload index with title](https://prepublish.ai/youtube-channel/bilawalsidhu).
- **"This 4D tech feels like a Time Machine"** — time/season comparison, described by the creator and mirrored in a [channel feed/archive](https://aitoolsreview.co.uk/channel/bilawal-sidhu). [Previously identified Short URL](https://www.youtube.com/shorts/_os_dFLezlg) (*the specific video ID could not be independently opened in this research session*).
- **Long-form memory-palace discussion:** [Previously identified video URL](https://www.youtube.com/watch?v=rVlRQakpOME) (*the specific video ID could not be independently opened in this research session*).
- The creator's description explicitly mentions **COLMAP SfM, MultiSet visual positioning (VPS), monocular depth, and pose-graph optimization**. See [channel feed](https://aitoolsreview.co.uk/channel/bilawal-sidhu) and [the creator's public post](https://www.linkedin.com/posts/bilawalsidhu_2d-moments-frozen-in-time-turned-into-3d-activity-7494939548670537728-PRRK).

### The crucial repo distinction

**God's Eye View** is the MIT-licensed, Cesium-based, evolving **3D globe/spatial-intelligence client**. It can be studied for rendering, geospatial integration and cinematic navigation. Its public README explicitly calls it an exploratory client, not a hardened production service. It is **not a verified public release of the full Memory Palace reconstruction pipeline**. Do not assume cloning it provides photo alignment, multi-year registration, or splat training.

To try the globe (follow current repo prerequisites; current contributing guide uses Node.js 24.14.x or 26.x):

```bash
git clone https://github.com/bilawalsidhu/gods-eye-view.git
cd gods-eye-view
npm install
npm run doctor
npm run dev
```

Keys are optional for a basic globe; Google/Cesium photorealistic providers have separate API/usage terms. See the [README](https://github.com/bilawalsidhu/gods-eye-view#readme), [contributing setup](https://github.com/bilawalsidhu/gods-eye-view/blob/main/CONTRIBUTING.md), [data-source terms](https://github.com/bilawalsidhu/gods-eye-view/blob/main/DATA_SOURCES.md), and [security model](https://github.com/bilawalsidhu/gods-eye-view/blob/main/SECURITY.md).

## 3. Important technical distinctions

| Term | Input | Output | What it does *not* guarantee |
|---|---|---|---|
| **Visual place recognition (VPR)** | One photo / embedding collection | Similar or likely same-place candidates | True overlap, same physical camera pose |
| **Local feature matching** | Pair of photographs | Corresponding keypoints and confidence | Valid 3D model unless geometry is verified |
| **Structure from Motion (SfM)** | Multiple views | Calibrated cameras + sparse 3D points | Photorealistic dense world or watertight mesh |
| **Multi-view stereo (MVS)** | Cameras + images | Dense cloud / surface evidence | Good reconstruction without sufficient overlap/texture |
| **Gaussian splatting (3DGS)** | Images + camera poses + initial scene | Photorealistic view-dependent 3D representation | Accurate physical surface, collision mesh, scale, consistent geometry |
| **NeRF** | Images + camera information | Neural radiance scene | Editable geometric mesh by default |
| **Visual positioning (VPS)** | A query photo and a known map | Camera location/orientation in a reference map | New full scene reconstruction |
| **Spatial memory / 4D timeline** | Scene, posed photos, date/time | Time-addressable spatial visualization | Reconstructing every date into complete independent 3D scene |

**Overlap matters more than image count.** A hundred near-identical photos can be worse for 3D than thirty overlapping but meaningfully separated viewpoints. Old and new pictures may be alignable to a common scene even if they are too sparse to reconstruct an entire year independently. GPS can narrow candidate searches but usually cannot align photos accurately to a building or object.

## 4. Finished applications: maturity, stack, deployment, quality

**Maturity legend (qualitative):** *Mature* = established and broad capabilities; *Established* = substantial project but still specialized; *Emerging* = newer or early releases; *Prototype* = useful ideas but limited production evidence. **Visual potential** assumes good input capture and is not a benchmark score.

| Project | Maturity | Stack / installation | Photo-to-3D | Outputs / visual potential | Fit for OpenProcessor |
|---|---|---|---|---|---|
| [WebODM](https://github.com/WebODM/WebODM) | **Mature** geospatial product | Docker/server; free Windows/macOS installers; web UI, API, processing nodes | **Yes** geospatial photogrammetry; current 2026 engines ODX/MicMac/LGT | Georeferenced map, point cloud, elevations, textured models; splat integration. **High for survey/terrain** | Best **durable backend UX/processing architecture reference**; not a general photo library |
| [Meshroom / AliceVision](https://github.com/alicevision/meshroom) | **Established** | Desktop node-graph UI; official Windows/Linux builds; no official macOS release | **Yes** | Detailed textured meshes, camera poses, depth. **High for well-captured objects/structures** | Study processing stages and debug UX; not a ready multiuser API |
| [OOOSplat](https://github.com/ooolabdev/ooosplat) | **Emerging**, 0.7.x | React + TypeScript + Tauri + Rust; `.exe`, Apple Silicon `.dmg` alpha, Ubuntu 24.04 `.deb` alpha | **Yes:** bundled COLMAP + Brush | Gaussian PLY, SPZ/SOG etc; offline viewer. **High photorealism potential** | Best downloadable one-click benchmark; **local Rust/Tauri app, not FastAPI/HTTP queue server** |
| [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) | **Advanced/evolving** | Native GPU desktop studio; Python plugins and MCP | **Yes** when provided appropriate camera/image data | High-end splats; PLY, SOG, SPZ, HTML; API lists other formats | Editing/inspection, export and automation reference; workstation, not gallery backend |
| [SuperSplat](https://github.com/playcanvas/supersplat) | **Established polished editor** | Browser TypeScript/PlayCanvas, [live editor](https://superspl.at/editor) | **No** photo-to-model training | Edit, clean, compress, share existing splats; PLY/SOG/SPZ | Strong UI + export/viewer reference; *not* reconstruction worker |
| [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) | **Established developer toolkit** | Python/CUDA CLI + interactive tools | **Yes** via datasets/conversion | NeRF/3DGS; very high visual potential | Algorithm/training benchmark; not a nontechnical photo-library product |
| [COLMAP OpenMVS App](https://github.com/yeicor/colmap-openmvs-app) | **Experimental small project** | Full-stack web + native Mac/Win/Linux bundles; containerized, separable backend; Dioxus/Rust app | **Yes** | Point clouds, textured meshes, GLB; high potential with good images | Good **simple full-stack/worker/viewer** reference; needs stability evaluation |
| [Atlas](https://github.com/nicholaskim17/atlas) | **Prototype** | Next.js 16/React 19, Firebase + KIRI Engine reconstruction API | **Yes, via external API** | Gaussian splats + SPZ; location/time journeys, audio, 3D walkthrough | Closest **consumer spatial-memory UX concept**, but cloud API/billing dependence; verify license |
| [Chronoscope](https://github.com/chronoscope-foundation/chronoscope) | **Emerging historical knowledge platform** | API-first temporal fact store, web map/time slider, ingestion/workers | **Not** general auto photo-to-splat engine | Historical locations, provenance and time-based evidence | Reference for temporal data, uncertainty and source provenance |
| [God's Eye View](https://github.com/bilawalsidhu/gods-eye-view) | **Evolving open-source client** | JavaScript/Vite/Cesium globe; optional APIs | **No verified Memory Palace reconstruction workflow** | Geospatial globe/terrain, data layers, cinematic visualization | Globe/navigation reference |
| [Immich](https://github.com/immich-app/immich) | **Mature photo-management benchmark** | Modern web/mobile, Docker self-hosting | **No documented automatic SfM/3D world pipeline** | Library, people, semantic search, map, chronological media | Competing gallery UX benchmark; optional integration concept |
| [LibrePhotos](https://github.com/LibrePhotos/librephotos) | **Established photo-library product** | React 19 + Django API + background tasks; single container, Compose, K8s | **No** documented scene reconstruction | Photo timeline, face clustering, image search, place grouping | **Excellent reference** for self-hosted photo UX and durable ingestion |

**Important 2026 correction:** WebODM currently states it has **decoupled from OpenDroneMap** and uses **ODX/MicMac/LGT**, not the old ODM-processing relationship. Its documented Gaussian splat path includes exporting training data, training with external software (e.g., OpenSplat/Brush), then uploading the splat for viewing. Do not assume every deployment generates splats end-to-end without an external training stage. Consult the [current WebODM README](https://github.com/WebODM/WebODM) and [splat workflow](https://docs.webodm.org/gaussian-splats/).

**Download/stability reality:** OOOSplat 0.7.x does have native Mac/Linux installers, but the project **explicitly labels those builds alpha**, with macOS unsigned/unnotarized and the macOS COLMAP stage CPU-only; Brush can train via Metal. Meshroom has no official macOS release. SuperSplat is an editor/viewer, not a training application. Software being downloadable does **not** automatically establish a stable multiuser server or unattended production processing service.

Sources: [OOOSplat English docs](https://github.com/ooolabdev/ooosplat/blob/main/README_EN.md), [releases](https://github.com/ooolabdev/ooosplat/releases), [WebODM downloads](https://webodm.org/download/), [Meshroom releases](https://github.com/alicevision/meshroom/releases), [COLMAP OpenMVS App README](https://github.com/yeicor/colmap-openmvs-app), [Atlas README](https://github.com/nicholaskim17/atlas), [Chronoscope README](https://github.com/chronoscope-foundation/chronoscope), [LibrePhotos README](https://github.com/LibrePhotos/librephotos).

## 5. Geometry/reconstruction engines

These are **background-worker engines**, not finished consumer applications.

| Engine | Maturity | Best at | Outputs | Automation / caveats |
|---|---|---|---|---|
| [COLMAP](https://github.com/colmap/colmap) | **Mature baseline** | General unordered-image SfM, camera poses, sparse/dense MVS | Cameras/intrinsics, points, dense cloud, surface/meshes | CLI + Python **pycolmap**, Docker; version 4.2 added global-mapper multi-component output and LoMa learned features |
| [OpenSfM 1.0](https://github.com/OpenSfM/OpenSfM) | **Re-established/active continuation** | SfM with GPS/geospatial references and mapping | Poses, sparse/dense clouds, textured mesh, georeferenced surfaces/maps | Python/C++; older [Mapillary repo](https://github.com/mapillary/OpenSfM) is no longer active |
| [openMVG](https://github.com/openMVG/openMVG) | **Established** | Modular feature geometry + SfM | Cameras, sparse points, downstream MVS interchange | Classical alternative, more plumbing for end-to-end |
| [OpenMVS](https://github.com/cdcseacave/openMVS) | **Established** | Dense reconstruction, mesh refinement/texturing after SfM | Dense clouds, textured meshes | **AGPL-3.0**, inspect implications before embedding in commercial product |
| [AliceVision](https://github.com/alicevision/AliceVision) | **Established** | End-to-end photogrammetry library behind Meshroom | Geometric/camera/dense/mesh outputs | Can run headless; GPU and dependency packaging need care |
| [GLUEMAP](https://github.com/colmap/gluemap) | **2026 research** | Difficult, low-overlap or repeated-structure reconstructions | COLMAP-compatible sparse model | SALAD retrieval → optional Doppelgangers++ → neural local geometry → global BA; dependency/checkpoint complexity |
| [VGGT](https://github.com/facebookresearch/vggt) | **Research, widely used** | Fast feedforward depth, point maps and pose estimates from few/many views | Predicted depth, cameras, tracks, COLMAP export | Promising fast preview; compare geometric accuracy; **weights have commercial-use restrictions** |
| [GTSfM](https://github.com/borglab/gtsfm) | **Research/developer toolkit** | Modular/parallel SfM | Camera poses + sparse geometry, supported downstream exports | Benchmark alternative; not a GUI photo app |
| [hloc](https://github.com/cvg/Hierarchical-Localization) | **Established academic toolbox** | Retrieval → local matching → 6DoF localization/scene SfM | Matched pairs, keypoints, camera localization | High-value **discovery/old-photo registration** component |
| [LightGlue](https://github.com/cvg/LightGlue) | **Established research component** | Fast learned matching of local descriptors | Image-pair correspondences | Pair matching only; follow with RANSAC/geometry; check chosen extractor/weights |
| [SALAD](https://github.com/serizba/salad) | **Research model** | Place-specific global image retrieval | Global place descriptor | Better complementary VPR feature than CLIP for place grouping; validate weights and deployment |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | **Established depth model** | Monocular depth where multi-view data sparse | Per-photo estimated depth | Useful for projections/preview, **not proof of physical geometry or global metric scale** |

**Why COLMAP first?** Widely used, reproducible, broad format compatibility, pycolmap integration, good diagnostic metrics, and strong conventional geometry when the dataset is suitable. **Why not just put every photo into COLMAP?** Multi-component mapping can split disconnected graphs, but matching tens of thousands of unrelated family photos would be wasteful and error-prone. Use OpenProcessor's retrieval/preselection to create smaller candidates first.

**GLUEMAP research architecture:** SALAD retrieval, optional repeated-structure pair filtering, Pi3/Pi3X/VGGT/MapAnything local backbones, global mapping/bundle adjustment, SIFT track refinement. Treat its multi-model dependency tree as optional experimental workers rather than core setup. [Project details](https://github.com/colmap/gluemap).

## 6. Gaussian splat trainers, formats, and viewers

| Tool | Role | Hardware | Main artifacts / trade-off |
|---|---|---|---|
| [gsplat](https://github.com/nerfstudio-project/gsplat) | CUDA rasterizer/training examples; **preferred NVIDIA backend** | NVIDIA/CUDA | Gaussian PLY through training pipelines; low-level library, not turnkey gallery |
| [Brush](https://github.com/ArthurBrussee/brush) | Cross-platform 3DGS engine | WebGPU, NVIDIA/AMD/Intel/Apple depending backend | COLMAP/Nerfstudio datasets → splats; very useful portable option; used by OOOSplat |
| [OpenSplat](https://github.com/WebODM/OpenSplat) | Portable C++ 3DGS trainer | CPU, CUDA, AMD, Metal (platform-dependent) | PLY, SPLAT, SPZ, RAD; CPU drastically slower; accepts COLMAP/OpenSfM/ODX |
| [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) | Interactive train/inspect/edit/export | Primarily NVIDIA GPU toolchain; consult current build/platform docs | PLY, SOG, SPZ, HTML, additional export API formats |
| [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) | Training/evaluation framework including Splatfacto/NeRF | Typically NVIDIA/CUDA | Novel-view images and trained scenes; solid research benchmark |
| [SuperSplat](https://github.com/playcanvas/supersplat) | **Editor and web publishing**, not trainer | Modern browser/WebGL/WebGPU | PLY, compressed PLY, SOG, SPZ and various imports |
| [PlayCanvas Engine](https://github.com/playcanvas/engine) | Real-time browser 3D viewer | Browser GPU | Good production candidate for interactive splat scene |
| [Spark](https://github.com/sparkjsdev/spark) | Three.js Gaussian rendering integration | Browser GPU | Convenient for three.js/Svelte ecosystems; compare large-scene performance |
| [CesiumJS](https://github.com/CesiumGS/cesium) | Global geographic/terrain 3D | Browser GPU + optional map providers | GPS/map/terrain representation; not a splat-training engine |

### Formats and what users should expect

| File / record | Represents | Suitable for |
|---|---|---|
| `cameras.bin`, `images.bin`, `points3D.bin` (COLMAP) | Sparse camera/point reconstruction | Canonical geometry interchange, quality measures and re-registration |
| Standard `.ply` | **Ambiguous:** mesh vertices/point cloud *or* Gaussian splat attributes | Preserve source model; validate schema before rendering |
| `.spz` | Compressed Gaussian splat | Efficient web distribution when viewer supports that version |
| `.sog` / related PlayCanvas bundles | Compressed/streamed splat | Browser viewing and potentially large-scene delivery |
| `.rad` | Gaussian scene variant used in WebODM workflow | WebODM-specific path; confirm viewer compatibility |
| `.glb` / `.gltf` | Mesh/scene asset (with textures/materials) | Games, surface geometry, web and 3D engines |
| Orthophoto / DEM / DSM / GeoTIFF | Georeferenced 2D elevation/map products | Aerial survey, GIS; not equivalent to a walkable photo scene |
| `scene.json` + `poses.json` (proposed) | OpenProcessor IDs, WGS84 origin/transform, cameras, timestamps, confidence, artifact pointers | Durable integration independent of trainer and viewer |

**Rendering distinction:** Splats tend to look more photographically realistic from captured/nearby viewpoints; textured meshes are better for collisions, measurements, surface editing, export to games, and geometry-dependent workflows. Neither is inherently superior for every scene. A single job may produce both.

References: [SuperSplat import/export support](https://github.com/playcanvas/developer-site/blob/main/docs/user-manual/supersplat/editor/import-export.md), [OpenSplat formats](https://github.com/WebODM/OpenSplat), [LichtFeld export reference](https://github.com/MrNeRF/LichtFeld-Studio/blob/master/docs/plugins/api-reference.md).

## 7. How the output quality actually compares

**There is no honest universal numerical ranking** without feeding *the same images* to each version with controlled settings/hardware and objectively measuring both geometry and appearance. The following are *capability potentials* and typical strengths, not measured project scores.

| Goal / imagery | Likely best starting route | Expected behavior | Main risk |
|---|---|---|---|
| High-fidelity room or building walkthrough, rich overlapping images | COLMAP → gsplat / Brush | **Excellent photorealism possible** | Floaters, blurry unseen angles, reflections, moving subjects |
| Accurate outdoor object/building surface | COLMAP → dense MVS / textured mesh | **High geometry potential**, often less photorealistic than splat | Textureless walls, windows, sky/vegetation |
| Georeferenced mountain road/overlook or aerial scan | OpenSfM / WebODM / COLMAP + GPS/ground-control | **High mapping potential** | Poor georeferencing if only imprecise EXIF GPS; long linear corridor drift |
| 5–20 random old photos of the same location | hloc registration to known scene; VGGT experiment | **Good chance of positioning some photos**; full 3D uncertain | Insufficient baseline, low overlap, changed environment |
| Many unrelated photo-library images | OpenProcessor discovery → per-scene reconstruction | No 3D until candidate set is validated | Semantic false positives / false clusters |
| Same place across years or seasons | Stable reference model + date-specific registered photos / layers | **Compelling spatial time travel**, often without separate yearly splats | Changing landmarks, missing camera metadata, temporal inconsistencies |

**Quality inputs:** overlap, camera movement/parallax, image sharpness, intrinsics, exposure consistency, scene reflectivity, occlusion, feature diversity, dynamic objects, and temporal change. A model cannot recover unseen sides of a building with classical photogrammetry; monocular/neural completions can guess, but guessed geometry should be labeled as inferred.

**Useful output profiles for the product:**

1. **Preview / sparse** — registered camera positions, sparse cloud, original photo markers. Very fast relative to dense training.
2. **Photorealistic world** — splat + compressed web asset + original camera pose metadata.
3. **Geometric model** — dense cloud/textured mesh; optional GIS outputs.
4. **Spatial memory only** — place/timeline + poses or coarse GPS anchors even if 3D reconstruction is inadequate.

## 8. What OpenProcessor already has — and the missing product layer

**Verified public repository:** [davidamacey/OpenProcessor](https://github.com/davidamacey/OpenProcessor).

From its published README, OpenProcessor already exposes:

- **FastAPI API** in front of **NVIDIA Triton/TensorRT** and **OpenSearch k-NN**.
- Image ingest: `/ingest`, `/ingest/batch`, `/ingest/directory`.
- Image/text MobileCLIP embeddings and semantic/vector search.
- Detection, OCR, face embeddings/search, and crop/object-level embeddings.
- Clustering and generated albums: `/clusters/train/{index}`, `/clusters/stats/{index}`, `/clusters/{index}/{id}`, `/clusters/albums`.
- Duplicate queries and health/monitoring; Docker Compose installation and GPU profiles.

These are documented features, **not an assertion that the existing clustering outputs are geometric place clusters**. Also, the publicly documented service architecture should not be interpreted as a complete durable 3D job system: verify job broker, persistence, retries, API contracts, frontend, permissions and storage implementation before integration.

| Capability | Public OpenProcessor baseline | New photogrammetry requirement |
|---|---|---|
| Image ingest | Yes | Capture/persist EXIF GPS, camera intrinsics, time and source provenance consistently |
| Semantic similarity | MobileCLIP | Add dedicated VPR descriptor and spatial candidate retrieval |
| General clustering | FAISS-based indices/albums | Distinguish image-topic clusters from **geometrically connected places** |
| Facial similarity | Yes | **Do not conflate a person's identity with a place**; permission/privacy handling |
| Local geometric matching | Not documented | Keypoints, pair matching, robust epipolar checks |
| Scene graph | Not documented | Persistent image nodes, verified edges, candidate scenes, versions |
| Reconstruction readiness | Not documented | Coverage, baseline, connectedness, predicted registration success |
| SfM/MVS | Not documented | Optional COLMAP/OpenSfM worker |
| Splat training | Not documented | gsplat/Brush/OpenSplat worker and GPU queue |
| Spatial memory/time viewer | Not documented | Timeline, photo pose anchors and date-specific overlays |

**Product design principle:** The application **suggests** scenes and automatically does inexpensive preflight analysis; expensive reconstruction is opt-in by default (automatic only for approved policies and budgets). The gallery stays useful even when a reconstruction fails.

## 9. Proposed automatic scene-discovery algorithm

This is the core differentiation. It should be evaluated as a *retrieval and graph-building problem*, not merely "run HDBSCAN again".

### 9.1 Offline ingest and descriptors

For each image, retain:

- Stable `asset_id`, owner/tenant, original asset URL, checksum/perceptual hash, dimensions, thumbnail.
- EXIF timestamp/time zone and uncertainty; GPS lat/lon/alt and **GPS accuracy/confidence**, camera model, focal length/lens data, EXIF orientation.
- Existing OpenProcessor MobileCLIP global and optional crop embeddings, tags, faces/objects.
- **Visual place recognition** embedding (e.g., SALAD), stored separately from semantic CLIP. Optionally multiple scale/crop descriptors.
- Flags for screenshot, duplicate, panorama, burst, privacy restrictions, low resolution, poor focus and dynamic foreground domination.

Keep GPS optional: older digital photos, scans, and exported assets often have no coordinates. Do not silently invent high-precision location from a rough landmark match.

### 9.2 Candidate retrieval — cheap, scalable

Use bounded retrieval, not all-pairs matching:

1. Propose neighbors by GPS radius if trustworthy, temporal sessions, similar album/event, VPR nearest neighbors, and **some** semantic CLIP neighbors.
2. Retain cross-year neighbors deliberately (otherwise temporal windows will miss historic images).
3. Exclude exact duplicates/bursts as independent reconstruction coverage, but retain one representative image and link the rest.
4. Limit top-K retrieved pairs and cap work per ingest/job; cache candidate edges per model version.
5. Apply tenant/album/permission boundaries before retrieving or evaluating photos.

### 9.3 Geometric verification — necessary gate

For top candidate pairs:

- Extract **SIFT / ALIKED** (or select COLMAP LoMa) local features; match with **LightGlue** or classical matcher.
- Use calibrated essential matrix or fundamental matrix RANSAC/MAGSAC; record inlier count and ratio, spatial distribution of features, likely parallax, degeneracy (planar/homography, low baseline), and uncertainty.
- Avoid a single global "overlap yes/no" threshold across all scenes; tune with labels and camera characteristics.
- Save an edge only if it passes geometric checks; classify relation as *verified overlapping*, *likely same-place, not geometrically verified*, or *unrelated/ambiguous*.
- Use place recognition/CLIP for **proposing pairs** and geometric verification for **confirming visible overlap**.

### 9.4 Build and refine a scene graph

- **Node:** photo and its features, capture metadata.
- **Edge:** verified geometric link with inlier statistics and relative geometry estimate.
- **Scene candidate:** robust connected subgraph (not simply a semantic cluster).
- Split spurious bridges and repetitive-pattern false connections using cycle/pose consistency; preserve separate physical spaces even when semantically identical (two similar hotel rooms).
- Detect partial overlaps and **possible merge edges between two prior clusters**. Revalidate with geometry before joining; avoid one giant component via weak transitive edges.
- Handle changes incrementally: newly ingested photo retrieves likely scenes, matches representative/keyframe images, updates graph; run bounded nightly repair/reclustering for ambiguous edges.
- Maintain stable `scene_id` with versioned membership so users' annotations do not disappear when clusters split/merge.

**Important distinction for your existing recursive clustering work:** Place graph refinement is different from embedding-space hierarchical clustering. Start with **geometry-consistent subgraphs** and use CLIP/HDBSCAN only for candidate indexing, semantic albums and graph partition proposals.

### 9.5 Reconstruction-readiness score

Store a detailed report, **not just a mysterious AI percentage**:

- Unique connected image count and percent of group with verified edges.
- Graph component/bridge/connectivity quality.
- Estimated baseline and viewpoint diversity, keypoint distribution.
- Preliminary camera registration success and reprojection error for selected subsets.
- Predicted scene coverage / isolated viewpoints.
- GPS/georeference availability and historical temporal coverage.
- Expected memory/storage/GPU cost and expected failure modes.

Labels: **High potential / Needs more viewpoints / Spatial-memory-only / Not enough evidence**. Learn thresholds from your benchmark dataset, not universal rules.

### 9.6 Reconstruction and continuous update

- On approval, freeze a **versioned scene manifest** of source image IDs and checksums.
- Run COLMAP SfM; store every registered camera and unregistered image with the reason/status.
- Optionally run dense MVS/mesh and/or gsplat/Brush training.
- Save artifacts and metadata separately; attach results to the existing gallery images rather than moving originals.
- New overlapping photos can be **localized into an existing scene** without retraining every model. Accumulate enough new coverage to propose a new version/retrain.

**Illustrative pseudocode (not running OpenProcessor code):**

```python
def on_asset_ingested(asset_id):
    asset = index_exif_semantic_and_place_descriptors(asset_id)
    candidates = retrieve_nearby_places_and_images(asset, max_pairs=100)
    for neighbor in candidates:
        if already_verified(asset_id, neighbor):
            continue
        geom = verify_local_features_with_ransac(asset, neighbor)
        persist_pair_evidence(asset_id, neighbor, geom)
    affected = update_scene_graph_and_versions(asset_id)
    for scene in affected:
        update_readiness_score(scene)
        maybe_suggest_reconstruction(scene)  # default: user-approved
```

The 100-pair cap is an **illustrative engineering parameter**, not a scientifically validated threshold.

## 10. Historical photographs and a four-dimensional memory timeline

### Design goal

User searches a place (e.g., cabin, building, overlook), switches **Photos / Map / 3D / Memory Timeline**, and scrubs through photographs captured at different times without leaving the library. The viewer preserves source images and provenance; 3D is a *view of photos*, not a replacement for them.

### Three practical levels

| Experience | Prerequisites | Quality/complexity |
|---|---|---|
| **Geo-photo timeline** | GPS + timestamps | Easiest; approx place markers; not camera-accurate |
| **Spatial photos in an existing 3D scene** | Reference reconstruction + feature-based camera localization | Best first **Memory Palace** milestone; accurately poses some historic photos |
| **Multiyear reconstructed worlds** | Dense overlapping imagery *per period* + cross-period alignment | Cinematic and expensive; may fail when imagery insufficient |

**Recommended path:** Construct a stable reference scene from the best recent photo set; then localize old photos to the scene using hloc/LightGlue, known landmarks, or optional VPS. If the historical photo cannot be localized, keep it on the timeline at GPS/estimated position with a visibly lower-confidence marker.

### Temporal complications

- Capture time can be missing, inaccurate or timezone-shifted. Store time precision/uncertainty.
- Architecture, roads, vegetation, furniture and weather can change; avoid treating two seasonal 3D scans as one static geometry without explicit date/layer separation.
- Dynamic people/vehicles should not necessarily become permanent reconstructed geometry.
- Cross-year cameras may not share reliable feature matches; allow user-provided anchor pairs/landmarks and manual transforms.
- Use spatial coordinates with explicit frame transforms: source camera coordinates → local scene frame → georeferenced WGS84/ENU/ECEF if trustworthy. Local SfM often has **unknown absolute scale**.
- A 2D image projected into 3D is not the same as reconstructing its hidden scene. Label projection/estimated depth appropriately.

**Viewer concept:** left photo gallery; right 3D scene (splat or mesh), colored camera frustums/photo planes; horizontal time slider; "2012 vs 2026" ghost overlay; click image to jump to its pose; switch to map, original media and spatial memory. The user chooses whether 3D is rendered. Consider a low-bandwidth thumbnail fallback.

**Commercial or hosted VPS:** [MultiSet](https://docs.multiset.ai/) is relevant to camera localization in an existing map, but pricing/service dependency and terms must be evaluated. It is **not a purely open-source substitute for COLMAP**.

## 11. Deployable architecture, storage, API, and workers

### 11.1 Service diagram

```text
Existing Svelte/Tailwind photo UI (proposed/integration layer)
    |  gallery / search / scene suggestions / map / 3D / timeline
    v
OpenProcessor FastAPI  +  auth & permissions
    |-- Existing ingest, Triton inference, OpenSearch embeddings
    |-- Scene candidate API / preview / status / metadata
    |-- Durable job creation, state/progress/events
    v
Durable broker / DB-backed job records (implementation choice)
    |-- CPU/cheap index+EXIF+VPR tasks
    |-- GPU feature matching/geometric verification tasks
    |-- Optional high-memory reconstruction workers
            |-- COLMAP (SfM + optional MVS)
            |-- GLUEMAP/VGGT experimental workers
            |-- gsplat/Brush/OpenSplat trainers
    v
Postgres: assets, scenes, image pairs, poses, artifacts, job states
OpenSearch: semantic and place embeddings + candidate retrieval
Object store / existing filesystem: originals, features, outputs
    |
    v
Viewer: PlayCanvas/Spark for splats, Cesium for geospatial world,
        optional glTF mesh viewer; timeline/registration metadata
```

**Recommended operational contract:** A single **top-level `docker compose up -d`** starts the existing application; an **opt-in `reconstruction` profile** adds heavy GPU workers. Docker Compose GPU flags, driver and toolkit support depend on host/Compose version. A Tauri desktop package is not automatically a remotely callable, multi-tenant service.

**No need to move all images:** Keep immutable original files wherever they already reside (NAS/MinIO/S3) and pass asset IDs/URLs into scene manifests. Minimize copies by staging only relevant, downscaled/matched working sets; keep full-res originals for reconstruction if permitted.

### 11.2 Durable jobs — requirements

- Persisted job records and statuses: `queued → preparing → matching → sfm → training/meshing → publishing → succeeded/failed/cancelled`.
- Idempotency key derived from scene version, pipeline version, source hashes, parameters and target outputs.
- Per-stage checkpoints and artifacts; retry after worker/container restart; cancellation with process-group cleanup; timeout and resource limits.
- One GPU-heavy scene job per allocated worker by default, with controllable parallelism and estimated VRAM/disk admission.
- Separate GPU inference (Triton) and reconstruction allocations so long-running 3D training does not block photo ingestion/search.
- Progress via server-sent events/WebSocket or polling; real logs and trustworthy stages (do not fake percent completion).
- User ownership/tenant boundaries and signed access to original photos, intermediate geometry and renders.
- Observability: queue wait, stage latency, GPU memory, throughput, failed registration rate, per-job storage and estimated cost.

**Avoid blindly mixing a Celery worker and its own orchestrator in one process.** Containerize COLMAP/3D processes in separate workers and use the application's chosen durable queue. FastAPI can submit/query job records; it should not directly run 30-minute GPU subprocesses in request handlers.

### 11.3 Proposed data model (illustrative)

```text
assets(id, tenant_id, source_uri, sha256, captured_at, time_precision,
       gps_lat, gps_lon, gps_accuracy, camera_model, focal_length, ...)
asset_descriptors(asset_id, descriptor_type, model_version, vector_ref, ...)
image_pairs(a_id, b_id, matcher_version, inliers, inlier_ratio,
            geometry_type, confidence, diagnostics_uri, status)
scenes(id, tenant_id, title, geom_frame, geo_transform, status, ...)
scene_versions(id, scene_id, manifest_hash, created_at, readiness_json, ...)
scene_images(scene_version_id, asset_id, role, registered, pose_id, ...)
camera_poses(id, scene_version_id, asset_id, R, t, intrinsics, covariance, ...)
scene_artifacts(id, scene_version_id, format, uri, checksum,
                bytes, pipeline_version, quality_json, ...)
reconstruction_jobs(id, scene_version_id, idempotency_key, state,
                    profile, worker_id, stage, progress_json, ...)
```

Use a **local Cartesian frame** for the reconstructed 3D scene, with an optional geospatial transform plus scale-confidence fields. PostGIS is useful for geographic candidate lookup and bounding regions, but the SfM camera/splat geometry should not be shoved wholesale into Postgres.

### 11.4 Proposed APIs (not existing endpoints)

Paths are written without a leading slash because none of them exists; the mount prefix is undecided.

```text
GET    scenes/candidates?status=high_potential
GET    scenes/{scene_id}
GET    scenes/{scene_id}/images
GET    scenes/{scene_id}/overlap-graph
GET    scenes/{scene_id}/readiness
POST   scenes/{scene_id}/reconstructions
GET    reconstructions/{job_id}
POST   reconstructions/{job_id}/cancel
GET    scenes/{scene_id}/artifacts
GET    scenes/{scene_id}/timeline
POST   scenes/{scene_id}/register-photo
POST   scenes/{scene_id}/manual-anchors
```

Use `POST` for state changes, explicit authorization, job idempotency, and versioned manifests. Default to the user's consent for private/biometric imagery; avoid unexpectedly reconstructing shared spaces from other users' assets.

### 11.5 Platform deployment

| Host | Practical starting use | Caveats |
|---|---|---|
| **Ubuntu x86_64 + NVIDIA** | Primary COLMAP + LightGlue + gsplat worker | NVIDIA driver/CUDA/VRAM/pinned images; test contention with Triton |
| **Apple Silicon Mac** | Gallery UI, ingestion, CPU SfM baseline; OOOSplat/Brush or OpenSplat Metal test | gsplat CUDA path does not run natively; OOOSplat COLMAP CPU-only in macOS alpha |
| **CPU-only Linux** | EXIF, semantic metadata, preview, limited SfM; optional OpenSplat CPU | Full 3DGS expensive; avoid automatic scheduling |
| **AWS GPU worker** | On-demand/Spot reconstruction with object-storage artifacts | Spot interruption, checkpoint resume, GPU costs and data privacy |

## 12. Model and engine selection

**Phase-one production default:** SALAD or alternate VPR descriptors → ALIKED/SIFT + LightGlue → COLMAP 4.2+ → gsplat → PlayCanvas.

Why this combination:

- **MobileCLIP alone** mixes *same subject category* with *same physical place*; add dedicated VPR and geometric verification.
- **SIFT/ALIKED + LightGlue** provides local correspondence; RANSAC checks physical consistency.
- **COLMAP** gives robust diagnostic camera geometry and interoperable outputs; its 4.2 multi-component handling helps but should not replace discovery.
- **gsplat** is suitable for NVIDIA automation, but it is a training/rasterization library, not one-click project UX.
- **SuperSplat / PlayCanvas** are mature presentation and editing references; keep training separate.

**Alternative engine policies:**

| Problem | Try next | Decision criterion |
|---|---|---|
| Low feature/overlap and difficult scene geometry | **GLUEMAP** with selected backbone | Additional registered views and lower error *on your benchmark*, acceptable complexity |
| Very few images; fast sparse preview | **VGGT** | Compare against COLMAP and label inferred vs measured geometry; license checkpoint |
| Geographic mapping/landscape and GPS-rich datasets | **OpenSfM 1.0 or WebODM processing** | Georeference/dense map output quality and operational ergonomics |
| No CUDA/Apple Silicon | **Brush or OpenSplat** | Runtime, output quality, memory, package complexity |
| Need physically useful mesh instead of just viewing | COLMAP dense / **OpenMVS** | Mesh coverage/texture/metric tests, plus AGPL decision |
| Need repeated historical photo localization | **hloc + LightGlue / VPS** | Correct camera registration in a fixed historical reference scene |

## 13. Quality gates, benchmarking, and failure modes

### 13.1 The evaluation dataset you actually need

Create one *deliberately mixed* library (e.g., 500–2,000 photos) with ground-truth groups:

- A walkaround of one building or exterior with good overlap.
- A similar building at a **different location** (semantic false-positive test).
- Indoor room(s), repetitive hallways, panoramas, trees, water, sky, glass.
- Same place photographed across seasons/years, with some major renovations.
- Old JPEGs/scans **without GPS** and inaccurate timestamps.
- Bursts/duplicates, close-ups/portraits that show almost no background.
- Negative controls: unrelated vacation and screenshot images.

Keep annotations for `same_place`, `geometric_overlap`, `same_scene_component`, `registerable_to_reference`, and `not_reconstructable`. Respect access permissions and avoid publishing private photos in benchmark outputs.

### 13.2 Score **discovery and reconstruction separately**

| Stage | Metrics |
|---|---|
| Candidate retrieval | Recall@K of valid image pairs, precision of retrieved same-place candidates, per-image cost |
| Geometry verification | Precision/recall of overlap edges; inlier count/distribution, false connections |
| Scene graph | Scene purity, coverage, fragmentation, erroneous merges, cluster-ID stability over ingest |
| SfM | Registered image percentage, median reprojection error, track length, camera pose accuracy where reference available |
| Georeferencing | Anchor residuals, landmark reprojection error, geospatial accuracy and uncertainty |
| Mesh | Coverage, completeness/accuracy to surveyed reference, hole/floating-surface rate |
| Photorealistic scene | Held-out PSNR/SSIM/LPIPS, human review, floaters, scene size and browser frame rate |
| Historical timeline | Fraction successfully localized, cross-year anchor drift, date correctness, user correction rate |
| Operations | Job failure/retry/recovery, elapsed time, GPU RAM, disk footprint, cost, memory leaks |

**Do not set universal registration/PSNR thresholds based on guesswork.** Establish acceptance criteria using a labelled set, visual review and user expectations. Reprojection error can be deceptively low even for wrong or incomplete scene geometry.

### 13.3 Expected failures

- CLIP clusters two visually similar but different places; **geometry gate prevents merge**.
- Two photos share the same sign/logo but no meaningful 3D baseline; reject weak overlap.
- A large set contains connected subgraphs joined by a few false edges; prune pose-inconsistent bridges.
- User has 20 photos from the **same exact viewpoint**; provide timeline, not promise full 3D.
- Old photo has no EXIF/GPS; retrieval and anchor-based registration may work, but keep uncertainty.
- Modern 3D reconstruction includes moving people or cars; mask/marginalize dynamic regions if necessary.
- Same site was remodeled; use separate temporal geometry layers or manually anchored comparison.
- Scenic outdoor shots of mountains with enormous distance and little parallax can look similar but be hard to reconstruct accurately.
- CUDA/driver/model versions conflict with Triton: pin worker images and isolate GPU resources.
- 3D files can be large: offer quality presets, compress to SPZ/SOG where supported, and retain source asset hashes.

## 14. Mac/Linux hands-on evaluation

**A. OOOSplat desktop (quick visual baseline).** Download a platform installer from [releases](https://github.com/ooolabdev/ooosplat/releases). Mac alpha requires macOS 15+ Apple Silicon; currently unsigned/unnotarized; Ubuntu alpha targets 24.04 x86_64. Feed it a deliberate set of **overlapping** photos or orbit/walkaround video. Compare high/low quality presets, camera registration, splat artifacts, visual floaters and export sizes. No remote FastAPI service comes with it.

**B. WebODM web service (durability reference).** Start with [official installers/Docker](https://webodm.org/download/) and [docs](https://docs.webodm.org/). Study user/project/task separation, job recovery, processing nodes, API and artifact viewing. Check latest 2026 ODX/MicMac/LGT configuration. Its [Gaussian workflow](https://docs.webodm.org/gaussian-splats/) documents training-data download → external splat training → upload; verify any newer integrated generation capability before assuming fully automated workflow.

**C. Meshroom desktop (traditional geometry baseline).** Download [Windows/Linux](https://github.com/alicevision/meshroom/releases); **no official macOS release**. Use it to inspect local features, SfM, depth, meshing and texturing stages; note CUDA requirements and failure diagnostics.

**D. COLMAP worker baseline.** Use the [official docs](https://colmap.github.io/) and pinned Docker/CLI version. An example for a coherent *single-scene* dataset:

```bash
# Example only. Confirm this syntax against the installed COLMAP version.
mkdir -p ./reconstruction
colmap automatic_reconstructor \
  --workspace_path ./reconstruction \
  --image_path ./photos
```

Record sparse camera registration and report failure causes *before* attempting splat training. The command does **not** autonomously crawl a mixed library and discover all good places.

**E. hloc/LightGlue prototype.** First verify predicted pairs and image correspondences, then RANSAC, then scene-graph grouping. Save match visualizations and manually audit false positives. [hloc README](https://github.com/cvg/Hierarchical-Localization), [LightGlue README](https://github.com/cvg/LightGlue).

**F. Atlas UI demo.** Its README documents a sample route at `/dev` without Firebase/KIRI keys, useful to evaluate its Svelte-equivalent consumer experience. Production reconstruction requires its external KIRI API: [Atlas repo](https://github.com/nicholaskim17/atlas).

## 15. Licensing, privacy, and dependencies

**Not legal advice; check exact repo commits, transitive dependencies, model checkpoints, datasets, and distribution rules.** The top-level repo license does not license every model weight or embedded dataset.

| Component | General licensing/review note |
|---|---|
| COLMAP | Permissive BSD-family; verify bundled dependencies and patents as applicable |
| OpenSfM | BSD-style in repo; verify new fork/third-party deps |
| GLUEMAP | Repo states BSD; **check optional Pi3/VGGT/MAST3R checkpoint terms** |
| LightGlue | Apache-2.0 code and main weights; **SuperPoint is separately restrictive**; ALIKED noted BSD-3-Clause |
| Brush | Apache-2.0 (repository LICENSE) |
| Nerfstudio / gsplat | Permissive open-source projects; verify target version and transitive deps |
| SuperSplat | MIT |
| WebODM | **AGPL-3.0**; code/service integration obligations need counsel-level review |
| OpenMVS | **AGPL-3.0**; review before bundling or modifying |
| VGGT | Original checkpoint **noncommercial**; separately approved `VGGT-1B-Commercial` weights allow certain commercial uses but **exclude military** and have specific terms |
| God's Eye View | MIT client; data providers and hosted/photorealistic tiles have independent terms |
| Atlas / experimental projects | Don't assume a license from a public repo; verify an explicit license and third-party service terms |
| Photographs and faces | Preserve user rights, authorization, retention, deletion and export controls; don't assume a place memory is consent to persistent biometric identity indexing |

Relevant primary references: [LightGlue licensing](https://github.com/cvg/LightGlue#license), [VGGT README](https://github.com/facebookresearch/vggt), [VGGT commercial model](https://huggingface.co/facebook/VGGT-1B-Commercial), [OpenMVS](https://github.com/cdcseacave/openMVS), [WebODM](https://github.com/WebODM/WebODM), [Brush LICENSE](https://github.com/ArthurBrussee/brush/blob/main/LICENSE).

**Architecture safety/privacy:** Keep scene discovery scoped to a tenant/user and explicit sharing permissions; make embedding/face models optional; avoid combining all users' photos to construct a shared space without an explicit feature and permissions model; support delete of source, derived features and all published 3D artifacts; protect EXIF GPS and camera trajectories.

## 16. Implementation phases and estimated effort

*Planning estimates for an experienced developer extending existing infrastructure; not project guarantees. End-to-end multiuser polish and arbitrary legacy image quality are the main schedule multipliers.*

| Phase | Deliverable | Approx. effort | Completion gate |
|---|---|---|---|
| **0. Benchmark foundations** | Curated mixed image library, labelled overlap/place pairs, COLMAP baseline | 2–5 days | Baseline metrics + known negative controls |
| **1. Automatic discovery** | EXIF + VPR descriptors, candidate retrieval, local geometry, scene graph | 1–3 weeks | High recall with low erroneous same-place merges |
| **2. Readiness + UI** | Scene suggestions, photo list, diagnostic overlap graph, user approve/ignore | 1–2 weeks | Useful suggested scene groups even without any 3D |
| **3. Durable reconstruction** | Optional Docker worker, jobs, retries, COLMAP artifacts, metadata | 1–2 weeks | Job survives restart and produces camera poses reliably |
| **4. Photorealistic viewer** | gsplat training + web export, embedded PlayCanvas/Svelte viewer | 1–2 weeks | User opens a scene from existing photo gallery |
| **5. Spatial timeline** | Register historic images, overlay by time, confidence markers and manual anchors | 2–4+ weeks | At least one real cross-year location navigable in 3D |
| **6. Production hardening** | Security, quotas, resource scheduling, multiuser isolation, performance, UX | 3–6+ weeks | Recovery, permissions, storage pressure and user testing |

The phases overlap and are not strictly additive. A useful **MVP can precede full 3D**: show automatically found scene groups and an evidence-based reconstruction readiness score. Multi-year alignment and hands-off reliability are more difficult than demonstrating one perfect Gaussian splat.

### Product milestones

- **MVP A — Smart photo discovery:** "We found these 128 images of the same cabin; 83 have verified overlap; 3D may be possible."
- **MVP B — One-click scene:** User clicks "Generate 3D"; durable job creates poses, splat and browser preview; originals remain indexed.
- **MVP C — Memory Palace:** An old photo at that location is localized and shown as a dated image plane or immersive projection; scrub years.
- **V2 — Continuous discovery:** Newly ingested images automatically attach to candidate scenes, prompt for retraining only when useful; add optional richer historical layers.

## 17. Complete research directory

### Inspiration and near-consumer experiences

- [Bilawal Sidhu — God's Eye View](https://github.com/bilawalsidhu/gods-eye-view) — open-source geospatial globe, not full published Memory Palace pipeline.
- [Bilawal Sidhu YouTube](https://www.youtube.com/@bilawalsidhu) — creator's videos/Shorts.
- [Memory Palace clip descriptions / video catalog](https://aitoolsreview.co.uk/channel/bilawal-sidhu) — 4D same-place/same-time descriptions.
- [MultiSet VPS docs](https://docs.multiset.ai/) — commercial visual positioning.
- [Atlas](https://github.com/nicholaskim17/atlas) — React/Next spatial journeys and KIRI-backed splats.
- [Chronoscope](https://github.com/chronoscope-foundation/chronoscope) — evidence-aware historical spatial timeline.
- [Immich](https://github.com/immich-app/immich) — photo gallery product/UX benchmark.
- [LibrePhotos](https://github.com/LibrePhotos/librephotos) — multiuser self-hosted gallery, similar image/face/background pipeline.

### Complete photogrammetry applications

- [WebODM](https://github.com/WebODM/WebODM), [download](https://webodm.org/download/), [current splat workflow](https://docs.webodm.org/gaussian-splats/).
- [Meshroom](https://github.com/alicevision/meshroom), [official builds](https://github.com/alicevision/meshroom/releases).
- [OOOSplat](https://github.com/ooolabdev/ooosplat), [English docs](https://github.com/ooolabdev/ooosplat/blob/main/README_EN.md), [installer releases](https://github.com/ooolabdev/ooosplat/releases).
- [COLMAP OpenMVS App](https://github.com/yeicor/colmap-openmvs-app), [releases](https://github.com/yeicor/colmap-openmvs-app/releases).
- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio).
- [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio).
- [SuperSplat](https://github.com/playcanvas/supersplat), [live editor](https://superspl.at/editor).

### SfM, geometry, matching and VPR

- [COLMAP](https://github.com/colmap/colmap), [documentation](https://colmap.github.io/), [version history](https://github.com/colmap/colmap/releases).
- [OpenSfM active project](https://github.com/OpenSfM/OpenSfM) and [old Mapillary upstream](https://github.com/mapillary/OpenSfM).
- [openMVG](https://github.com/openMVG/openMVG).
- [OpenMVS](https://github.com/cdcseacave/openMVS).
- [AliceVision](https://github.com/alicevision/AliceVision).
- [GLUEMAP / CVPR 2026](https://github.com/colmap/gluemap).
- [VGGT](https://github.com/facebookresearch/vggt).
- [GTSfM](https://github.com/borglab/gtsfm).
- [hloc — Hierarchical Localization](https://github.com/cvg/Hierarchical-Localization).
- [LightGlue](https://github.com/cvg/LightGlue).
- [SALAD place retrieval](https://github.com/serizba/salad).
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).

### 3D splatting, viewers and formats

- [gsplat](https://github.com/nerfstudio-project/gsplat).
- [Brush](https://github.com/ArthurBrussee/brush).
- [OpenSplat](https://github.com/WebODM/OpenSplat).
- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio).
- [PlayCanvas Engine](https://github.com/playcanvas/engine).
- [SuperSplat format reference](https://github.com/playcanvas/developer-site/blob/main/docs/user-manual/supersplat/editor/import-export.md).
- [Spark for Three.js](https://github.com/sparkjsdev/spark).
- [CesiumJS](https://github.com/CesiumGS/cesium).

### Your code and extension points

- [OpenProcessor — public GitHub repository](https://github.com/davidamacey/OpenProcessor) — documented FastAPI/Triton/OpenSearch, image ingestion, MobileCLIP/face/OCR, FAISS clustering, Docker deployment.
- [OpenProcessor installation](https://github.com/davidamacey/OpenProcessor/blob/main/INSTALLATION.md) — confirm current version and hardware profiles before setting up.

## 18. Open questions and recommended first experiment

Questions worth resolving *with experiments rather than assumptions*:

1. Is **SALAD** materially better than existing MobileCLIP on the user's actual same-place/changed-season photos? How much does it cost at ingest time?
2. What feature/matcher combination is most efficient: SIFT/ALIKED+LightGlue versus COLMAP's newer LoMa? What is pair-verification precision on repetitive architecture?
3. Can an image graph identify separate viewpoints of one cabin across years **without** merging visually similar sites?
4. Does COLMAP outperform GLUEMAP/VGGT on few-view, non-orbit, uncalibrated personal-photo sets once training/compute cost is included?
5. Is **gsplat vs Brush vs OpenSplat** best for your hardware, deployment size and actual photo quality?
6. What percentage of suggested scenes are truly good enough for users to approve 3D, and how much GPU/storage does each require?
7. Can old photos be reliably registered against a **new reference scene** without training a second complete model?
8. Should the photo UX stay in existing OpenProcessor Svelte integration, or interoperate with a mature gallery like Immich/LibrePhotos? Avoid rebuilding unrelated gallery capabilities without a product need.
9. Which reconstruction tools and checkpoints have licenses compatible with the intended AGPL/open-source and commercial distribution strategy?

### Recommended experiment — one weekend / initial proof of concept

**Dataset:** ~500–2,000 mixed photos, with 100–300 known overlapping views of one place; add unrelated and visually similar-location distractors; include a few dated historical photos. Ground-truth which pairs and groups are truly same-place and overlapping.

**Test:**

1. Export existing OpenProcessor embeddings and image IDs; compute one dedicated VPR embedding for the same data.
2. Retrieve candidate neighbors from each embedding space; measure same-place recall@K.
3. Verify candidate pairs with ALIKED/SIFT + LightGlue + robust geometry; manually review a sample of false positives.
4. Build connected scene candidates; score viable groups; record split/merge errors.
5. Feed the **best discovered group** (not manually assembled) to COLMAP. Compare recovered poses and registration against a manually curated ground-truth collection.
6. Train one gsplat scene from the poses, export a browser-compatible asset, and prototype a Svelte photo/3D toggle.
7. Register several historical images to the resulting scene and scrub through dates with confidence labels.

**Go/no-go:** OpenProcessor reliably identifies correct same-place groups and avoids false geometry-based merges. If it cannot, focus on better retrieval/matching/graph quality before spending months polishing a 3D viewer.

---

## Closing recommendation

The easiest impressive demo is a Gaussian splat of one manually collected orbit. The **more defensible product capability** is continuous, permission-aware discovery of spatially connected images **inside an existing photo library**, with reconstruction treated as one possible derived representation of a scene. A timeline then gives a useful experience even when 3D cannot be reconstructed from every period. Keep the photo application modular: cheap retrieval always on, geometry verification bounded, expensive 3D workers opt-in, and Svelte viewers consuming stable artifacts and pose metadata.

*This file is research and planning documentation, not deployed software. Repository capabilities, licenses and installers can change; follow the linked primary documentation before implementation.*
