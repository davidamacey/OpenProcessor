# 13: backend speed plan (image processing, GPU serving, clustering and labelling)

| Field | Value |
|---|---|
| Status | plan, revision 2 (independent review of revision 1 applied); items are coded batch by batch as each is decided and reviewed |
| Code base | `int/b4-stateless` at `d08d5e7a0` |
| Replaces | the open parts of `06b_b5_decode_once.md`, `06c_b5_crops_ocr_faces.md` and `06d_b5_processing_plans.md` (their keep-lists move here when a batch merges) |
| Issues | umbrella #500; #493, #494, #495, #496, #499, #501, #502, #503, #504, #505, #507, #508, #509, #510, #67, #346 |
| Hardware | two RTX A6000 (48 GB each): GPU 0 serves Triton today, GPU 2 is free |
| Numbers | every number names its dataset and run; raw outputs are kept outside the tracked tree and summarised in section 10 |

## 1. What this plan is for

The user's job ([VISION_AND_GOALS.md](../../VISION_AND_GOALS.md)): load photos, let an existing
detector find the objects, label whole clusters in the user's own classes, train a model, serve it.
Two request shapes set the targets, because they stress different parts of the system:

- **A single-branch pass over a large library**, for example faces only: the other models are off,
  so decode, prep and the small models decide the speed, and grouping must stay correct as photos
  are added.
- **A full pass at bulk scale**: detect, embed every wanted crop, cluster, label by cluster, train
  and serve. The embedding encoder decides the speed. The target is 100 photos per second.

Everything below is judged by measured photos per second, p50 and p99 latency, GPU use and memory
on the running stack, with output parity against the stored references and no loss of cluster
quality.

## 1a. Focus and order (owner direction, 2026-11-14)

The goal, in one sentence: **one photo goes in, is decoded once on the GPU, every model result is
produced on the GPU from that one frame in one pass, the results are stored once and compactly, and
they come back to a search or a grid page at once.** Model speed in isolation is not the goal; the
time from "photo in" to "everything stored and findable" is. Nothing in today's stack is fixed:
the model server, the vector store and the engine language are choices to be made by measurement.

What the code does today, read from the source (not from its tests):

| Path | What happens to one photo | Source |
|---|---|---|
| `/analyze`, all branches | The encoded file is handed to four separate paths on a thread pool: detector + whole-image embedding (CPU decode and CPU preprocessing), faces (CPU decode, CPU box decoding and alignment, two model-server hops), OCR (its own decode, one recogniser call per text crop), crop embeddings (decoded again, cropped on the CPU). Four decodes of the same file | `src/services/analyze_pipeline.py` |
| Curation ingest | One GPU decode serves the detector and the PE frame and crop vectors; every crop's pixels then return to the host to be JPEG-encoded on CPU threads. Faces are not on this path. A different embedding family from the core routes | `models/gpu_image_prep/`, `src/services/curation/ingest_gpu_decode.py` |
| Region embedding | The source is decoded again on the CPU for every item | `src/services/curation/region_embed_stage.py` |
| Thumbnails | Re-decoded from the original | `src/services/curation/image_serving.py` |
| Storage | Several PostgreSQL transactions per photo; an update of any machine result re-indexes the document's vectors | section 2, D5 |

Order of work from here. Each phase ends in the same headline row: **one 20 MP photo with every
branch on, photos per second per card and milliseconds per stage**, with accuracy checked against
ground truth (detection AP, face verification, text accuracy, retrieval recall), never only
against the previous output.

1. **The frame engine with every branch** (E1, then F1, O1b's rule, classification on crops, T0):
   decode once; detect, frame and crop embeddings, faces with alignment cut on the GPU from the
   full-resolution frame, OCR with batched recognition, stored thumbnail and crops; one request,
   one plan. The Triton model server stays behind it for now.
2. **One write per batch** (P1 and task 7): results leave the engine once and are written as a
   coalesced batch; no per-photo transactions, no barrier between a batch's GPU calls and its
   writes.
3. **Storage layout and the vector store** (D5, V0, V1, T8): machine results that change live
   apart from vectors; vector width and precision chosen by measurement (half floats, 8-bit,
   1-bit with rescoring, a learned projection to fewer dimensions) on at least one million real
   vectors; tuned OpenSearch against Milvus, Qdrant, PostgreSQL vector extensions and a
   Faiss or cuVS index with PostgreSQL bookkeeping; a flat vector file written at ingest so
   clustering never reads through the search engine.
4. **Grouping** (F3, L2, T7): faces, objects and whole images, each a rebuildable index apart
   from detections and from human decisions.
5. **Recall and display** (L3, L6, T0): search and grid pages with stated latency bars at one
   million photos.
6. **The engine we own** (T1, raised from a measured track to a build): a Rust engine behind
   protocol version 1 with decode, GPU kernels and TensorRT execution in one process and no model
   server between prep and inference, built on the owner's existing Rust serving code. It
   replaces the Python engine when it holds parity and wins the headline row; the protocol, the
   parity harness and the branch designs carry over unchanged.

Open questions this order has to answer early, by measurement: what share of a photo's time is
decode at full resolution and whether a reduced decode can serve the fixed-size model inputs
while full resolution is kept only for face, text and small-object crops; how many crop
embeddings a photo really needs and whether one encoder family can serve both the core routes
and curation; whether crop vectors can be pooled from one pass over the frame instead of one
pass per crop.

## 2. What binds today (evidence)

Baseline rows are in section 10. The findings that shape the design:

1. **The embedding encoder is the arithmetic bound when every crop is embedded.** The Perception
   Encoder (PE-Core-L14-336, FP16) is 72 to 91% of the GPU kernel time per photo and serves about
   200 to 225 inferences per second on one A6000 (224.6 at batch 16 through Triton). At 4.67
   crops per photo plus the frame (5.67 inferences) that is 35 to 40 photos per second for the
   encoder alone, and about 29 to 36 once the other kernels are counted. At 2.41 crops per photo
   the stack reached 48.3 photos per second with the GPU 94% busy.
2. **Below the ceiling, the GPU waits for the host.** On the 4,000-image mix the GPU is 65% busy
   at 22 to 26 photos per second; with one client on 20 MP photos it is 16% busy. The batch route
   runs in phases (all GPU calls of a request, then all writes, then an index refresh). Inside the
   prep model a photo spends 74 ms in the detector call (about 2.4 ms of GPU work) and 124 ms in
   the encoder call (about 24 ms of GPU work): queue delay (25 ms and 15 ms) and waiting for batch
   partners.
3. **Faces and OCR do not use the resident frame.** `/faces` decodes on the host and aligns on the
   CPU: 1.8 to 3.1 ms of GPU work per image against 0.36 to 0.43 s of API CPU per 16 MP photo.
   `/analyze` decodes the same bytes four to five times. OCR calls the recogniser once per text
   crop although the engine takes batches of 64.
4. **Decode and prep run inside Triton's Python backend, calling other models from 16 threads.**
   That configuration is not documented, and the crash it produced once (SIGSEGV in
   `libtriton_python.so`, every model down) matches two upstream issues that are open and unfixed
   through Triton 26.09 (triton-inference-server/server #8882 and #7517).
5. **The labelling loop pays for writes it does not need.** A 10,000-item label operation waits
   for an index refresh on each 500-item page (a floor of about 20 s). Every cluster refresh
   rewrites every member document, and every such update re-indexes the document's vector. List
   pages spend about 1.8 ms per row in Python serialisation, four times the search itself. The
   "photos containing X" join is cut at 10,000 photos and its negative form is wrong past the cut.
6. **Semantic search spends 75% of its time encoding the query on one CPU instance** (about 53 ms
   p50, 148 ms p99).

### 2.1 The path to 100 photos per second

No single item reaches the target. The factors multiply, and each has its own evidence and gate:

| Factor | Evidence today | Item | Expected |
|---|---|---|---|
| Encoder inferences per photo | 5.67 at policy `all`; 3.41 when only the wanted classes are embedded gave 48.3 photos per second on one card | P6 (embedding policy as a throughput setting) | 1.3 to 1.7x on the encoder-bound pass, with a quality gate |
| Cards | 1.91x measured for the encoder on two A6000 | P5 | about 1.9x |
| Pipeline fill | GPU 65% busy on the mix, 94% in the best configuration | P1, P2 | up to 1.4x on the mix; little where the GPU is already full |
| Encoder speed | FP16 near the published best for this network; an INT8 recipe that failed vector parity was 1.47x | T3 | 1.0 to 1.4x, only if a mixed-precision recipe passes every gate |

On one card with every crop embedded the honest ceiling is about 35 photos per second. Two cards
with the wanted-class policy reach about 90 on today's numbers; reaching 100 needs either P1 and
P2 to close the last fill gap or a win from T3. The ceiling itself is checked first: the
per-engine curve of the encoder at batch 16 and 32 under saturation, and the encoder's share of
GPU time at 4.67 crops per photo. If the encoder serves more than about 250 inferences per second
the table is redone.

## 3. Design decisions

Each decision names its evidence and the measurement that can overturn it.

### D1. Decode and prep leave Triton's Python backend; which host is decided by spike E0

**Decided 2026-11-14 from E0 (rows in section 10): arm B, a standalone engine process, with CUDA
shared memory as its transport and the plain request body as the second, explicit transport
setting.** System shared memory is not built.

- Speed does not separate the arms: on 20 MP photos one card gives 44 to 47 photos per second
  in every arm and the card is 84 to 93% busy. Arm B with CUDA shared memory is 1.02x, 0.97x and
  0.96x of arm A at one, two and four processes, with 15% less host CPU per photo.
- Arm B passed the kill test on both shared-memory transports (50 of 50 kills each, the model
  server never restarted). Arm C did not: after its Python stub was killed the prep model stayed
  not ready for at least 45 s with no recovery by itself. Arm A is today's path, where the same
  night a model unload hung and only a restart of the whole model server cleared it.
- Transport. System shared memory is 0.94x of arm A at two processes and needs the engine in
  the model server's IPC namespace, so the engine container has to be recreated after every
  model-server restart. The request body needs no coupling at all and costs 2% at one process
  and 9% at two. CUDA shared memory is the only transport at 0.95x or better at every process
  count. The rule below about beating system shared memory by more than the spread is not met
  (the ranges overlap); CUDA shared memory is chosen on the 0.95x rule and on operations.
- What E0 changed in the rules below: `pid: host` is not needed. Registration fails only when the
  engine process has the same process id as `tritonserver` (both are 1 by default), so the
  engine container runs with `init: true`. A killed engine leaves its regions registered and
  1 GiB of GPU memory pinned until the next engine start unregisters them by prefix; that step
  is mandatory. The model server needs `--allow-client-shm=true`.
- The gap to arm A at two and four processes is batch fill at the encoder (8 to 12 crops per
  batch against 17 to 19), not transport. E1 sends crops of every photo in flight through one
  queue per process; more than one engine process per card is a measured setting, not a default
  (the +51% for a second process was measured under the scan regression and is withdrawn).
- Not measured in E0 and measured on the E1 engine instead: arm C throughput, the unary front,
  small images, 16 and 64 photos in flight.

Decode, EXIF orientation, letterbox, crops, face alignment, OCR crop handling and the order in
which models run are one component, the **frame engine**. It keeps today's design (decode once,
keep the full-resolution frame on the GPU, cut every model input from that frame). What changes
is where it runs and how it reaches the TensorRT models. The requirement is fixed: **a fault in
decode or prep must not take the TensorRT models down.** The host is chosen by E0 from three arms
measured on the same photos, the same engines and the same day:

| Arm | Host | Calls the models by | Keeps for free | Open risk |
|---|---|---|---|---|
| A | today: Python backend model in the one Triton | BLS from worker threads | everything | the crash class takes every model down |
| C | a second `tritonserver` that hosts only the prep model | Triton client | model control, metrics, the front | the Python-backend stub is still there, but its fault ends only the prep server |
| B | a standalone process with a KServe v2 unary front | Triton client | nothing; needs its own readiness, metrics and config answers | new code to own |

For arms B and C the model inputs are small (1.2 MB for the detector letterbox, 339 KB per encoder
square), so E0 also measures three transports: CUDA shared memory, system shared memory, and the
plain request body
([shared-memory extension, 26.10](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2610/user-guide/docs/protocol/extension_shared_memory.html)).

Rules for the choice:

- An arm must pass the kill test: its process is killed at least 50 times with 32 photos in
  flight, and the model server stays up, logs no CUDA error, returns to its memory baseline and
  serves the next request.
- CUDA shared memory is used only if it beats system shared memory by more than the run-to-run
  spread. It has two costs the others do not: the exporting process must outlive every import
  (CUDA documents freeing exported memory before the importer closes it as undefined behaviour),
  and the two containers must share a PID namespace because the handle carries the exporter's
  process id. If it is used, regions are plain device allocations made once at start, registered
  once, addressed by offset, unregistered by name prefix when a new engine process starts, and
  both containers run with `pid: host`.
- Arm B is chosen over arm C only if it wins by more than the spread or C fails the kill test.
- The winner must reach at least 0.95x the photos per second of arm A at equal process count and
  equal crops per photo, with p99 and host CPU per photo not worse, and pass the parity profiles
  (`scripts/bench/parity.py`) read against a control captured twice, because the TensorRT detector
  is not bit-stable between runs. Blur sums and crop JPEG bytes must be identical.

What is decided regardless of the arm:

- **Engine processes per card are a measured setting.** One process saturates a card on 20 MP
  photos; when more are configured the API spreads requests over them with the existing router
  (`src/clients/triton_router.py`).
- **Overload is an answer, not a fallback.** A full engine answers with a retryable status; the
  API retries with backoff. No silent second route.
- **Engine down means the image fails visibly** (its decode route says so and the bulk run retries
  it). The API keeps no pixel-decoding fallback for ingest.
- **The request protocol is version 1 below**, written once before the engine is built, so later
  items are API-side work.

Why not a Triton ensemble: its data flow is static, so one photo cannot fan out to a
data-dependent number of crops. Why not the DALI backend: it lost 19 to 40% on 12 to 24 MP photos
in our earlier measurement and failed the letterbox parity gate.

#### Engine protocol, version 1

One request per image, KServe v2 tensors.

| Input | Type | Meaning |
|---|---|---|
| `image` | bytes | the encoded file (any format the engine decodes) |
| `plan` | one JSON string | see below |
| `boxes` | float64 `[n, 4]`, optional | stored boxes: the detections come from the caller, not the detector (reprocess); float64 because the stored `bbox_norm` values are Python floats and float32 could move a rounded pixel edge of the cut |

`plan` fields: `branches` (from `detect`, `embed_frame`, `embed_crops`, `faces`, `face_embed`,
`ocr`); `detector` (model name, input size, letterbox fill); `embed` (model name and preprocessing
recipe for the frame and for crops, so the core routes can ask for MobileCLIP2 and curation for
PE); `embed_classes` (detector class ids that get a crop vector; absent means all);
`if_detected` (class ids per conditional branch); `faces` (detector input size or tiling, stored
score floor); `crops` (`inline` or `none`); `derivatives` (named stored renditions such as a
thumbnail: longest side and quality).

| Output | Present when | Content |
|---|---|---|
| `status`, `decode_route` | always | `ok` or the reason the image failed; which decoder produced the pixels (`gpu`, or `none` when nothing decoded) |
| `branch_status` | always | one JSON string, one entry per requested branch: `ok`, `error` with a reason, or `not_run` (condition false); the schema is in `src/domain/frame_engine_wire.py` |
| `size` | decoded | width and height after orientation |
| `det_boxes`, `det_scores`, `det_classes`, `letterbox` | `detect` | as today |
| `whole_embedding` | `embed_frame` | one vector |
| `crop_embeddings`, `crop_embedded` | `embed_crops` | vectors and the mask of detections that got one |
| `blur_sums` | `detect` | as today |
| `crop_jpegs` | `crops: inline` | as today |
| `face_boxes`, `face_landmarks`, `face_scores`, `face_quality` | `faces` | every face above the stored floor; quality columns are face side in pixels, blur, and a pose measure from the five points |
| `face_embeddings` | `face_embed` | one vector per face |
| `ocr_quads`, `ocr_texts`, `ocr_scores` | `ocr` | per text region |
| `derivative_jpegs` | `derivatives` | one JPEG per named rendition |

### D2. A request is a plan; the engine runs only the requested branches (#67)

`/analyze`, core `/ingest` and curation ingest resolve a plan through
`src/domain/processing_plan.py` and send it to the engine. Decisions on the open questions of
`06d_b5_processing_plans.md` section 3:

| ID | Decision | Reason |
|---|---|---|
| Q1 | The legacy selection flags are removed in the same change. | No shims (repo standard); the frontend has no consumer. |
| Q2 | One decoder feeds every branch; the face and OCR references are re-captured after the gates pass on public data. | Four to five decodes per `/analyze` request today. Gates: ArcFace cosine p01 at least 0.995 against today's path on the same faces; OCR exact-text share at least 0.98. |
| Q3 | A curation request may narrow the project's policy, never widen it (422). | The policy is the project's contract. |
| Q4 | The `selected` mask is resolved by class name in the API and sent as `embed_classes`. | Class identity is the name; one parity test against `ingest_policy`. |
| Q5 | A failed branch returns 200 with that branch marked `error`; 502 only when every branch failed. | One failing model must not discard the other results. |
| Q6 | A missing dependency is refused (`missing_dependency`, 422). | Fail closed. |
| Q7 | Core `/ingest` without `embed_frame` stores no global vector; `near_duplicates` requires `embed_frame`. | No zero vectors in an index. |
| Q8 | No named presets. | Nothing needs them yet. |
| Q9 | A reprocess scope is a plan with stored boxes. | One executor instead of two. |

Every pixel consumer and where it goes:

| Consumer of decoded pixels today | After |
|---|---|
| Curation batch ingest, GPU route and CPU route | engine (the engine picks the decoder per image; the size gate is an engine setting, P4) |
| Single-image ingest and dataset import | engine (import uses stored boxes) |
| A project with a secondary detector | engine: the second detector is a second `detect` model in the plan; until that lands such a project keeps today's route and the old code is deleted with it, not before |
| Region-embed stage (decodes the source again per item, 217 to 220 ms per item) | engine with stored boxes (E3) |
| Core `/detect`, `/embed`, `/search/image`, `/faces`, `/ocr`, `/analyze`, core `/ingest` | engine through plans (A1) |
| Crop-cache miss and thumbnails | stay on the host decoder: one small image per user request, not a pipeline |
| Training export and VLM crops | read stored crops; unchanged |

`OP_INGEST_GPU_DECODE` and `OP_INGEST_GPU_MIN_MEGAPIXELS` are retired in `retired_env.py` when
the last API-side decode for ingest is deleted.

### D3. The batch route streams

Each photo of a batch request moves on its own from the engine call to its write. The writes stay
coalesced: the process-wide writer (`ingest_writer.py`) already batches documents across photos
and requests, and it stays. The index is refreshed by its refresh interval, with one explicit
refresh when a run or request ends only where a caller reads its own writes. The work in flight is
bounded by the engine and the writer, not by the request shape. Database connections per request
are sized from the timeline measurement, with a bar of no `database_busy` at 64 clients.

### D4. Faces are a project stage

This supersedes the "core-only now" recommendation for decision O1 in
`cluster_search_makemodel_plan.md`: grouping one library's faces is a per-library job, and the
project is the unit of isolation.

- A project owns a `faces` index role beside `images` and `items`. A face row is an immutable
  fact: box, five landmarks, detector score, quality measures, the embedding, and the versions of
  the models that produced it. Every face above a low stored score floor is kept; thresholds apply
  at read time, so changing one never means running a model again.
- People, memberships and every human decision (confirm, reject, merge, split, cannot-link) are
  rows in PostgreSQL, apart from the machine grouping and keyed by a stable face id, so a
  re-grouping never loses a decision. When a detector or recogniser changes, new detections are
  matched to stored faces by box overlap so the id and its decisions carry over.
- Grouping is a rebuildable projection of the face rows and the decisions.
- The detector input scale is a measured setting: a 640 letterbox shrinks a 40-pixel face on a
  24 MP photo to a few pixels. F1 measures 640, 1280 and tiles on the full-resolution frame by
  face-size bucket.
- "Every photo of this person" is the person's accepted members from PostgreSQL plus a radial
  search (a minimum score, not a top-k:
  [radial search](https://docs.opensearch.org/latest/vector-search/specialized-operations/radial-search-knn/))
  around the person's prototypes for review candidates.
- The deployment-wide faces index holds names, merges and locks today. They are exported into the
  new decision tables before the core routes are removed (F4); nothing is deleted without that
  step.

### D5. Machine results that change must not re-index vectors

Any update of an OpenSearch document re-indexes the whole document, its vector included
([update API](https://docs.opensearch.org/latest/api-reference/document-apis/update-document/)).
A cluster refresh and a label operation therefore pay for vector indexing today. Writing only the
changed items (L2) reduces how often; only a split of the item document into an immutable vector
document and a small mutable one removes it. The split is measured first (V0), before L2 and L6
are written, and every index change of this plan (the split if adopted, the shard count, the
`faces` role, the class-name field of L4) ships as **one migration wave**: one reindex per
project.

### D6. Precision and batching are chosen per model by measurement

FP16 stays where it is. The text tower moves from ONNX Runtime FP32 on the CPU to TensorRT FP16
(S1). INT8 is adopted for a model only when it passes that model's quality gate and wins at least
1.15x on the engine (section 8). Queue delays, preferred batch sizes and instance counts are set
from the per-engine curves and the photo-mix sweep (P2).

### D7. A card is a unit: one model server and its engines

Triton cannot pin a request to the card its input lives on. So each card runs its own Triton with
its own engine processes, and the API spreads requests over the cards' engines with the existing
router. A second card is a second copy of the same pair; no per-card model names. Profiles decide
what a card runs: bulk ingest runs the pair on both cards; labelling gives GPU 2 to the VLM and
the segmenter. Switching a profile under load is part of the soak (R1). One card remains a
supported target (the 12 GB acceptance, #510, is unchanged).

## 4. Work items, ranked

Each item: baseline, documented approach, check, expected win, risk. "Measure first" items have a
spike before code.

### Batch 1: independent wins (no dependency on the engine)

| ID | Item | Baseline | Approach (docs) | Check | Expected win | Risk |
|---|---|---|---|---|---|---|
| S1 | PE text tower on TensorRT FP16 (#493) | 53 ms p50, 148 ms p99 per uncached query, one CPU instance | Built as the image tower is built (FP16 baked into the ONNX, TensorRT 11 strong typing; [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html)); the export refuses an engine that fails the gate | Cosine to the FP32 tower on 2,000 COCO captions: median at least 0.999, p01 at least 0.995; batched equals batch 1; text-to-image recall@10 within 0.003 | Query encode under 10 ms | Low |
| R2 | Per-photo decode route in the ingest record | Only a process-wide counter and summed stage seconds | One closed vocabulary of routes; `decode_route` on the ingest result and the image document; `decode_routes` on the bulk run | Every route decision has a test; a status outside the vocabulary fails a test | Enabler for every decode measurement | Low |
| L1 | Label operations refresh once | A 10,000-item label operation: 20 pages, each waiting for a refresh (about 1 s) | `refresh=false` on page writes and one refresh before the job reports complete ([bulk API](https://docs.opensearch.org/latest/api-reference/document-apis/bulk/)); synchronous single-item writes keep `wait_for` | `label_op_probe.py` lost and extra writes stay 0; undo unchanged; the lock rule untouched | 10,000 items: about 20 s to a few seconds | Low; a write path, reviewed by the coordinator |
| L3 | List pages cost what a page costs | About 200 ms per 100 rows (1.8 ms per row in Python); an exact total and a second aggregation search on every page; the frontend pager re-copies every loaded row per page | Totals and hidden counts on the first page only, carried in the cursor; one serializer made cheaper with byte-identical output; the pager appends in place | A cursor page issues one search and no aggregation; serializer output identical on 1,000 documents; **browser time to row 10,000** and per-page API time by depth measured before and after | 3x or more per page; linear deep scroll | Low |
| O1b | OCR recognition in batches | One recogniser call per text crop; 42 to 46 ms of pipeline time for about 9 ms of GPU | **Spike only** (the orchestration moves into the engine, so no product code lands in the Python backend): crops sorted by aspect ratio and resized to a small set of fixed widths so a batch needs no padding (padded batches changed text: 86% exact); PaddleOCR's own batch recipe as the comparison arm | Character accuracy against the ground-truth text of a public scene-text set not below batch 1 (the first gate, exact-text share against batch 1 of at least 0.98, was withdrawn: it scored a candidate against today's output, not against the truth; section 10) | 2 to 4x on the OCR branch | Medium (text parity) |
| I1 | Idle cost of the worker fleet | About two cores with nothing running: the periodic worker 83% of a core, the export worker 54%, PostgreSQL 43% (`docker stats`, 2026-11-14). Causes: each container health check starts a full Celery process every 30 s (17 CPU-seconds per probe on the export worker); the 5-second ticks build an event loop and clients per project; registry polls scan the projects table about 350 times a second | The health probe asks the worker through Celery's own inspect API from a task-less app ([Celery workers guide](https://docs.celeryq.dev/en/stable/userguide/workers.html)); one client set per tick; the polls replaced by a change check | Every check the probe makes today still fails when the pool hangs; tick cadence unchanged | About two cores returned to ingest and to every measurement | Low |
| R3 | The API recovers when the model server restarts | After a restart the model server was ready in 29 s and the prep model loaded in 4 s, but every API worker answered 503 for 122 s more (gRPC reconnect backoff on the long-lived channels) | Bounded reconnect backoff through the channel arguments the client documents ([gRPC connection backoff](https://grpc.github.io/grpc/core/md_doc_connection-backoff.html)) | Restart the model server under a light load: health returns within 15 s of its gRPC port opening | Minutes of failed requests after every restart | Low |
| P0 | Ingest finishes a batch in a fixed number of transactions | A batch of 32 photos opened about 193 database sessions: four short transactions and 6.3 to 6.5 connection checkouts per photo, each commit waiting 40 to 60 ms for the disk | One claim, one placement and one record step per batch; advisory locks taken in key order in one statement ([explicit locking](https://www.postgresql.org/docs/18/explicit-locking.html)); class names created per batch; capture metadata read by a small pool of kept-open ExifTool processes ([ExifTool `-stay_open`](https://exiftool.org/exiftool_pod.html)) | Same documents and rows as before for the same input; two batches claiming the same content at once; the timeline re-run | The finish phase was 78 to 87% of a request | Medium: the claim transaction, reviewed by the coordinator |
| P0s | Database, search and broker volumes on solid-state storage | The measured stack keeps them on a spinning-disk array: a commit costs 40 to 60 ms | The installer names the storage requirement and warns when a data path is on a rotational device (`/sys/block/<dev>/queue/rotational`); the benchmark stack moves its volumes to NVMe (copied, originals kept) | The ingest rows measured four ways: old and new code, spinning and solid-state | Separates the storage effect from the code effect | Low |
| V2 | A photo never exceeds the nested-document limit | An item with 75 or more boxes reads back wrong per-box vectors, and a partial or scripted update then stores them (k-NN issue 3623, OpenSearch 3.6.0 and 3.9.0 with derived source) | Interim: every writer refuses a write that would take an item to 150 nested documents (74 boxes with one box-vector space, 49 with two; `src/domain/nested_limit.py`), the region and labelling drains skip such an item, a finder script lists existing ones. The structural fix (per-box vectors out of the nested field) is decided with V0 | The repro of section 10 stays at 0 wrong at the cap on the pinned version | Correctness | Low for the cap |

### Batch 2: the pipeline

| ID | Item | Baseline | Approach | Check | Expected win | Risk |
|---|---|---|---|---|---|---|
| E0 | Spike: the three host arms and three transports | arm A on the same day | D1 | D1's rules | Decides D1 | Throwaway |
| E1 | The frame engine on the chosen host, protocol version 1 | as E0 | D1. Decode through nvImageCodec where it matches or beats the hand-written nvJPEG lanes on the same photos (it adds progressive and CMYK JPEG and other formats; [nvImageCodec](https://docs.nvidia.com/cuda/nvimagecodec/)); resize kernels stay unless CV-CUDA passes the tensor gate at no more GPU time ([CV-CUDA](https://cvcuda.github.io/CV-CUDA/)) | `tests/gpu` parity tier, the stored prep and decode references, R1 | Robustness; one route for every image | High: GPU memory hand-off; core written by the coordinator, wiring by lanes |
| R1 | Soak: a decode fault cannot take the models down | One crash in about five bulk runs, not reproduced since | Large public photos at concurrency 16, 32 and 64 with model unload and load between runs, the engine killed on purpose mid-run, and a profile switch under load; today's path soaked once for its rate | Model-server restart count unchanged; every photo ingested after the bulk run's retry; crashes per 10,000 photos reported | The requirement of item 0 | Low |
| P1 | Streaming batch route (D3) | 32 photos per request: about 4.3 s, GPU 16% busy at one client; the wall-clock timeline of one request names the stages | D3 | Same documents as today for the same input (store parity); `failed` 0; no `database_busy` at 64 clients | Measured: the gap between GPU 65% and 94% busy on the mix | Medium |
| P2 | Batching and instances from measurement (#502) | Detector delay 25 ms, encoder delay 15 ms and one instance, mean encoder batch 21 of 32 | Per-engine curves with `perf_analyzer`, then a sweep of delay, preferred sizes and instance counts on the photo mix ([dynamic batcher](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)) | Config only: parity profiles pass | 5 to 20% | Low |
| P3 | Crop JPEGs off the critical path (#505) | 52 ms of host CPU per 20 MP photo on encode threads | *Measure first:* the timeline shows whether the response waits for the encode. If it does: `crops: none` at ingest by policy, with the existing miss route cutting a crop on first read | Crop bytes identical on the miss route; first-open latency p95 not worse | 0 to 15% | Medium |
| P4 | The size gate, re-measured | 3 MP, set before the current encoder layout | Ingest sweep at 1, 2, 3, 5 and 8 MP, GPU and CPU decode inside the engine | Parity per route | Correct routing | Low |
| P5 | Second card for bulk ingest (D7) | One card | A second Triton and engine pair on GPU 2 behind the router | Parity profiles per card | About 1.9x on the encoder-bound pass | Low |
| P6 | Embedding policy as a throughput setting | Policy `all`: 5.67 encoder inferences per photo; the policy modes were timed on 50 images only | Measure `all`, `selected` (wanted classes; minimum box side and score) and `lazy` on the 4,000-image mix; the default stays `all` unless the gate passes | Text-to-image recall@10 and the cluster metrics within 0.01 of `all` for the wanted classes, real photos | 1.3 to 1.7x on the encoder-bound pass | Medium: a quality decision |
| E3 | Region-embed stage on stored boxes | 217 to 220 ms per item (the source is decoded again per item) | The engine's stored-box input | PE vectors within the parity gates against today's CPU-cut vectors | About 2.5x on that stage | Low |
| E4 | The engine is the image path on every profile | Only the 40 GB profile turns the GPU decode on; the three smaller profiles run the engine at its smallest sizes (16 MiB region, 64 MiB frame budget, one worker) and never call it, so a 24 GB card decodes on the CPU | Engine sizes per profile from measured memory (frame budget for at least one 50 MP photo, a region for one encoder batch), the model list of the `ingest` load profile, floors re-derived per profile | Each profile's stack starts inside its card's memory and ingests the mix with `failed` 0; photos per second per profile | One decode on the GPU for every user, not only on 48 GB cards | Medium: the 8 GB floor is an owner decision |

### Batch 3: plans and faces

| ID | Item | Baseline | Approach | Check | Expected win | Risk |
|---|---|---|---|---|---|---|
| A1 | Plans on `/analyze`, core `/ingest` and curation ingest (D2) | `/analyze` 55 photos per second without OCR and 20 with it (COCO, 16 clients); four to five decodes | The routes resolve a plan and call the engine; legacy flags removed; per-branch outcomes | One skip-proof test per branch; default plans reproduce today's responses (the core routes ask for their own detector and embedder in the plan) | A single-branch request pays only for its branch | Medium |
| F0 | Face scoring harness | Nothing scored | Verification on LFW, CALFW and CPLFW; identity clustering on the pooled sets; error by quality signal; stability when photos are added in batches | The harness's own unit tests | The baseline every face change is judged by | Low |
| F1 | Face branch on the resident frame | 36 photos per second on photos at 16 clients, host-bound | SCRFD on a letterbox or tiles cut on the GPU (640, 1280 and tiles measured by face-size bucket); anchor decode and NMS; five-point similarity alignment cut from the full-resolution frame on the GPU; ArcFace in batches | Boxes within 1 px at 640; ArcFace cosine p01 at least 0.995 against today's path; verification accuracy on the three sets not lower; recall by face-size bucket reported | 3 to 6x on photos for a faces-only plan; more small faces found | Medium |
| F2 | Faces in the project (D4) | Faces only in one deployment-wide index | The `faces` role in the migration wave of D5 (the PostgreSQL role check, the index for every existing project, the combine, clone, delete and capacity paths); the people and decision tables | Project isolation tests; the guard refuses a foreign faces index | Faces usable per library | Medium (migration) |
| F3 | Face grouping rebuilt against F0's scores | Single linkage at cosine 0.7 | Quality-gated core points, incremental assignment to per-person prototypes, a review band, same-photo cannot-link, human decisions as constraints | F0's bars; real photos only | High recall at high purity; additions never regroup the library | High: written by the coordinator |
| F4 | People routes on the project | Core routes on the global index | Project routes for people, the review queue and "every photo of this person" (D4); existing names, merges and locks exported into the decision tables; then the core routes are removed | Contract tests; recall of one identity at fixed precision | The face workload end to end | Medium |
| T0 | Stored thumbnail at ingest | Thumbnails re-decoded from the original, with a per-process cache | A `derivatives` entry in the plan; the engine cuts the rendition from the resident frame | Thumbnail bytes decode to within the stored-crop quality bar; list latency p95 | A grid page never decodes an original | Low |

### Batch 4: clustering and scale

| ID | Item | Baseline | Approach | Check | Expected win | Risk |
|---|---|---|---|---|---|---|
| V0 | Spike: item document split and update cost | Every update re-indexes the vector | Time 100,000 label and membership updates on documents with and without the vector field; filtered k-NN with the filter fields in a second document is not possible, so the spike also lists which filters must stay beside the vector | Decides D5's layout | Decides L2, L6, V1 | Throwaway |
| L2 | Cluster projection writes only what changed | Every member rewritten on every refresh | "Changed" is membership and state only; score and margin are read from PostgreSQL; the out-of-order guard stays on the writes that remain; stage timers for snapshot, fit, carry-over, record and projection | Projection result identical to a full write | Refresh write time proportional to change | Medium |
| L4 | "Photos containing X" without the 10,000 cap | A `terms` aggregation cut at 10,000 image ids; the negative form is wrong past the cut | The class names of a photo's items kept on the image document (which class field, and how the image-counts refresh keeps it consistent with label writes, is written before coding); ad hoc item filters page with a `composite` aggregation ([composite](https://docs.opensearch.org/latest/aggregations/bucket/composite/)); the field ships in the migration wave | 12,000 matching photos: exact total, paging reaches the last photo; the negative form exact at 10,001; staleness after a label write bounded and stated | Correctness at scale | Medium |
| L5 | One vector reader | `_source` float lists through one sequential cursor; seven readers with their own paging | OpenSearch 3.7.0 first (the stack pinned 3.6.0; vector retrieval through `docvalue_fields` arrived in 3.7), then one pager and one decoder: sliced point-in-time pages, the vector as base64 float32 decoded per page ([docvalue_fields](https://opensearch.org/blog/retrieve-vectors-5x-faster-with-docvalue_fields-in-opensearch/)); nested per-box vectors keep `_source` through the same pager until the docs cover them; every caller keeps its own filter | Vectors bitwise identical per caller; the sweep passes on 3.7.0 | About 5x on the fetch | Low for the reader; the version bump is verified by the sweep |
| L6 | Cluster cards and outlier order from stored results | One aggregation with about ten sub-aggregations per page; outlier order cached per process | Per-cluster counts and outlier order kept in PostgreSQL by the projection and the label operations | Counts equal to the aggregation on a seeded project | Card list independent of library size; no per-process cache | Medium |
| V1 | Shards and vector storage | One shard per index | Section 7 | Section 7's bars | Scale | Medium |

## 5. Measured tracks (spikes that end in a decision, not a default)

| ID | Track | Question | Decision rule |
|---|---|---|---|
| T1 | Rust engine behind protocol version 1 | Does an engine with decode, prep and inference in one process beat the chosen arm? | Compared per plan (detect-only, faces-only, full) on throughput by image size, p50 and p99, GPU busy, memory, image size, start time and parity. A port is recommended only where it wins by more than the spread on a plan that matters; a full port is an owner decision. |
| T2 | Slim Triton | What do a TensorRT-only image and the removal of PyTorch save? | Built with Triton's `compose.py` ([compose](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/compose.md)); adopted if every model loads and the parity tier passes. |
| T3 | Encoder cost | Can the crop encoder cost less at equal cluster quality (INT8 mixed precision; a smaller encoder for crops)? | Section 8 gates; cluster quality on labelled vehicle and fine-grained sets, real photos only. |
| T4 | VLM serving | Which vLLM settings and which model label mini-clusters fastest at equal accuracy? | Label agreement at least 0.97 against today's answers on 1,000 public crops; crops per second. |
| T5 | Segmenter | What does SAM 3 cost per image and per prompt on this card, and does one image pass serve all prompts? | Baseline first; a change needs identical hits. |
| T6 | Training and serving | Export build time, trainer data loading, promote time, serving settings of a promoted model | Baseline; fix what the baseline shows. |
| T7 | Clustering on ground truth | Does a different cluster-count rule or method lower labelling decisions per item at equal precision? Does a GPU fit agree with the CPU fit? | The protocol of `09c_research_clustering_protocol.md`; five seeds; above the seed floor. A GPU fit is adopted only when L2's stage timers show the fit dominating. |
| T8 | Vector stores | How do Milvus, Qdrant and pgvector compare with tuned OpenSearch on the same 1 million real vectors? | The bars of section 7 and a 100,000-item membership rewrite under 60 s. |
| T9 | Ingest metadata | What does capture-metadata extraction cost per photo at ingest? | Measured in the timeline; reported with the ingest numbers. |

## 6. Not in this plan, with the reason

| Item | Decision |
|---|---|
| CPU letterbox in the API (#495) | Dropped: the API stops preparing pixels for ingest (D2); small images are decoded on the CPU inside the engine and letterboxed on the GPU. |
| Asynchronous OCR (#496) | Covered by plans: a caller that does not want to wait leaves `ocr` out and runs it later as a reprocess plan (Q9). No second mechanism. |
| Progressive JPEG guard (#494) | Covered by E1: the engine decodes progressive files itself (CPU, or GPU through nvImageCodec if that wins). |
| MPS (#507) | Not pursued: it needs a root daemon and a fatal fault under MPS reaches every client. Reopened only if E0 shows time-slicing between processes costs more than 10%. |
| EXIF-upright frame vectors for copies whose orientation tag was dropped | Not pursued: such files hold rotated pixels and no tag, so there is nothing to upright; frames with a tag are already embedded upright. Matching rotated copies would need four encoder passes per frame. |
| HEIC and RAW at ingest | After E1: a decoder in the engine; the codec licences are an owner decision. |
| Research scripts that need `leidenalg` or `igraph` | Deleted with their always-skipped tests in the Batch 4 clean-up unless T7 adopts a graph method, which then pins its own research tier. Recorded results stay. |
| Remote GPU index build | Not applicable: it serves Faiss fields, ours are Lucene. |

## 7. Vector storage at library scale

Sizing at today's dimensions: one million photos with 13 crops each is about 13 million 1,024-d
vectors, about 55 to 60 GB in a Lucene shard, above the usual 10 to 50 GB per shard. Decisions to
take by measurement on real, non-replicated public vectors (1 million first):

- shard count for the items index (1, 2, 4, 8) with concurrent segment search;
- Lucene scalar quantisation (4-bit, 1-bit) and on-disk mode with rescoring against float:
  recall@10 within 0.01 under no filter and under 2.6% and 32% filters, p99 under 100 ms;
- the document split of D5 (V0);
- the embedding policy (P6), which also decides how many vectors are stored.

## 8. Precision per model

| Model | Today | Candidate | Gate |
|---|---|---|---|
| YOLO11 and YOLO26 small detectors | FP16, uint8 input | INT8 with Q/DQ (TensorRT ModelOpt) | COCO AP and AP50 within 0.5 points; the `precision` parity profile; at least 1.15x on the engine |
| PE image encoder | FP16 | INT8 mixed precision (SmoothQuant or per-layer search); the all-INT8 recipes failed vector parity | Every `precision` gate; cluster purity within 0.5 points; at least 1.15x |
| PE text encoder | ONNX Runtime FP32 on the CPU | TensorRT FP16 (S1) | S1 |
| MobileCLIP2 image and text | FP16 with some FP32 layers | keep | none |
| SCRFD, ArcFace | FP16 | keep FP16; INT8 only after F0's scores exist | Verification accuracy within 0.2 points; cosine p01 at least 0.98 |
| PP-OCRv6 detection and recognition | FP16 | keep; batching first (O1b) | O1b |
| SAM 3 | FP32 weights with BF16 autocast | T5 | T5 |
| VLM | BF16 | weight-only quantised checkpoints that Ampere supports | T4 |

## 9. Order, lanes and sweeps

1. Batch 1 lanes run now; they touch no engine code.
2. The baseline rows, the per-engine curves, the request timeline and E0 run first on the GPU.
   E1 starts when E0 has chosen the arm; protocol version 1 does not change after that.
3. P1, P6, E3 and A1 are API-side work against protocol version 1 and follow E1. F0 has no
   dependency and runs now; F1 is an engine branch.
4. V0 runs before L2 and L6; the migration wave carries every index change at once.
5. The tracks run beside the batches as GPU time allows.

Each batch: lanes in their own worktrees with light checks, merged into `int/b4-stateless`, one
sweep, then the same benchmark rows as the baseline. A change is kept only if it wins and holds
parity; a dropped item is recorded in section 10 with its numbers.

## 10. Results

Measured 2026-11-13 and 2026-11-14 on the running stack unless a row says otherwise.

| Row | Dataset | Before | After | Parity | Decision |
|---|---|---|---|---|---|
| Encoder ceiling (`perf_analyzer`, FP16 engine, one instance) | synthetic input | assumed 200 to 225 inferences per second | 225 measured, compute-bound at 4.4 ms per image; batch size above 8 and concurrency above 2 add nothing | not applicable | The ceiling of section 2.1 stands |
| Detector ceiling (`perf_analyzer`, two instances) | synthetic input | not measured | 1,037 images per second at batch 16, not yet saturated | not applicable | The detector is not a bound |
| Queue delay at one request | synthetic input | not measured | encoder: 24 ms p50, 15 ms of it queue delay; detector: 45 ms p50, 25 ms of it queue delay | not applicable | Input to P2 |
| End-of-image scan in the prep model (regression of 2026-11-13) | 4,000-image mix, 4 clients | 21.9 photos per second on 2026-11-11; 10.6 with the regression (three runs, 0 failed) | fixed in `432674aa0`: the scan takes about 2 ms instead of about 80 ms per 7 MB photo and releases the interpreter lock; 16 threads scan 1,315 photos per second instead of 7.4 | identical end offsets on 400 real files and on generated streams | Kept; the re-measured mix row follows |
| S1 parity: PE text tower, TensorRT FP16 against ONNX Runtime FP32 | 2,400 COCO val2017 captions | gate: median at least 0.999, p01 at least 0.995 | cosine median 0.999979, p01 0.999950, min 0.999780; batched against batch 1 within 1.3e-5 | passes | Latency row follows |
| F0: recogniser as served | LFW, CALFW, CPLFW (10-fold) | not measured | accuracy 0.9965, 0.9575, 0.9558; the detector found no face in 17 LFW and 100 CPLFW pairs | not applicable | Baseline for F1 and F3 |
| F0: grouping, pooled set of 36,926 faces (BCubed) | LFW, CALFW and CPLFW joined by identity | single linkage at 0.7: precision 0.996, recall 0.407 | at precision 0.99 or better: single linkage at 0.55 recall 0.785, average linkage at 0.4 recall 0.889, a label-propagation graph method recall 0.896 | not applicable | F3 replaces single linkage; the quality gate as first designed lowered precision and recall and is not built; naive batch-by-batch assignment fell to precision 0.845, so incremental assignment needs its own design |
| Mix after the scan fix, 4 clients | 4,000-image mix (`op-bench-v1`) | 10.6 photos per second with the regression; 21.9 to 22.7 on 2026-11-11 | 18.1 and 19.2 (0 failed, host load 11 to 19); the prep stages are as fast as on 11-11 or faster (prep queue 138 to 161 ms per photo against 189 to 294) and GPU 0 is 52 to 57% busy | not applicable | The gap is on the API side; since 11-11 each batch shares two PostgreSQL connections for three short transactions per photo. The request timeline decides |
| Mix after the scan fix, 8 clients | same | 10.9 with the regression | INVALID: 503 after 22 s, the PostgreSQL pool (16 per process, no overflow, 5 s wait) ran out | not applicable | A finding for P1: database work per batch, not per photo |
| E0: prep host and transport, one process (direct to the model server, 32 in flight, 1,500 requests per point, 3 rounds, 0 failed) | set P, 20 MP | arm A (Python backend, one instance) 44.4 photos per second | standalone process: CUDA shared memory 45.4 (1.02x), system shared memory 44.8 (1.01x), request body 43.4 (0.98x); host CPU per photo 0.19 to 0.21 s against 0.23 | boxes, scores and classes identical on 200 photos; crop bytes identical; vector cosine at least 0.9995 | Speed is not what separates the arms: the card is 84 to 93% busy. D1 is decided by the kill tests |
| E0: two processes | set P, 20 MP | arm A with two instances 47.4 | standalone, two processes: 45.9 (0.97x), 44.7, 42.9; arm A fills encoder batches better (16.6 against 8.9 to 11.7) | as above | One card gives 44 to 47 photos per second on 20 MP photos at policy `all`; more needs P5, P6 or T3 |
| S1 latency: PE text tower | synthetic tokens, engine side (`trtexec`), not through the model server | ONNX Runtime CPU: batch 1 46.1 ms p50, 73.0 ms p99; 42 queries per second at batch 8 | TensorRT FP16: batch 1 1.37 ms p50, 1.38 ms p99; 2,778 queries per second at batch 8, 3,609 at batch 32; plan 681 MiB, built in 59 s | S1 parity row | Kept; the request-level number follows the sweep. Open: the plan has no build stamp, so a changed recipe is not detected |
| F3 prototype: average linkage resumed from the existing groups | pooled set of 36,926 faces (BCubed) | single linkage at 0.7: precision 0.996, recall 0.407; F0's batch assignment: precision 0.845 | cosine 0.40: one shot 0.990 / 0.889; 10 batches 0.990 / 0.896; 100 batches 0.990 / 0.898; the same groups as scipy average linkage on LFW and bed 3 (adjusted Rand index 1.0); 12.6 s for the pooled set | not applicable | F3 is built on this: group sum vectors, best-first merges, existing groups never split |
| F3 prototype: review band, simulated reviewer | same | automatic at 0.45: recall 0.844 at precision 0.991 | questions between 0.30 and 0.45: recall 0.948 at precision 0.991 for 3,167 questions (93% answered yes); below 0.30 five times the questions for 0.015 of recall | not applicable | Band 0.30 to 0.45 is the starting setting, re-measured on project data |
| Ingest request timeline on a quiet host (one client, wall clock per stage) | set P, 20 MP, and the 4,000-image mix | not measured | the finish phase (database and search writes) is 78 to 87% of a request: a single 20 MP photo takes 5.07 s, of which 1.04 s is the engine and 3.97 s the finish; four short transactions and 6.3 to 6.5 connection checkouts per photo; a commit waits 40 to 60 ms for the write-ahead log; capture metadata costs 54 ms per 6.7 MB photo, single-threaded; 213 ms per photo between the client and the first stage at one client | not applicable | The stack's database volumes are on a spinning-disk array. Two fixes, measured apart: P0 (code) and P0s (storage) |
| P0: ingest finish per batch | unit tier; capture metadata on generated JPEG, PNG, TIFF, WebP, AVIF, GIF and corrupt files | about 193 database sessions per batch of 32; 32 ms of ExifTool per 14 to 17 MB photo | at most 5 sessions per batch at any batch size; 9 ms per photo; identical tags | 125 unit tests; the database tier and the concurrent-claim test wait for the sweep; HEIC not tested | Merged (`bf7e5298c`); the ingest rows follow the stack refresh |
| O1b spike, round 1: recognition batching inside today's pipeline | public scene-text set, ground-truth text | per crop 21.6 ms, of which 10.2 ms is the batcher's queue delay and 4.5 ms the GPU; about 152 ms of recognition per image; the card 6% busy | exact-width groups: bit-identical text, about 6% faster (6.6 crops and 6.1 distinct widths per image); padded batches: character accuracy 0.5522 today against 0.5706 with the PaddleOCR batch recipe | padded batches are not worse against the truth | The first gate (0.98 exact-text share against batch 1) is withdrawn; round 2 measures the padded recipe and the queue delay |
| k-NN issue 3623 reproduced on the item index body | generated items with 10 to 80 boxes, OpenSearch 3.6.0 and 3.9.0 | not known | 0 wrong at 10, 40 and 74 boxes; at 75 and 80 boxes 96 of 100 items read wrong per-box vectors (99 after a merge); after the app's update shapes the nested k-NN is wrong in 300 of 300 queries; the frame vector is never wrong; with derived source off 0 wrong, at 20.5 KB per vector (3.6.0) and 41 KB (3.9.0) against 4.2 KB | not applicable | V2: cap at 74 now; the structural fix with V0 |
| E1: the frame engine against the prep model it replaces | 434 photos (public sets and set P), direct and through the API | the Triton Python-backend prep model | detections identical on 424 of 424 decoded photos, blur sums identical, 1,959 of 1,959 crop JPEGs byte-identical, vector cosine at least 0.999889 (the run-to-run floor of the encoder), stored-box mode equal; 50 of 50 engine kills survived with the model server untouched; after a model-server restart the engine re-registers by itself and serves again 27.9 s later; idle device memory 3,340 MiB at the 40 GB sizes and 348 MiB at the smallest | passes | Kept; every model call has a 15 s deadline and every request a 60 s one. The sweep owes the full suites, the image scan and the end-to-end parity script |
| I1, L5, S1 as merged | | | merged without a request-level number: the idle cost, the vector reader (OpenSearch 3.9.0) and the text tower through the model server are measured at the stack refresh | unit tiers pass | Rows follow the refresh |
