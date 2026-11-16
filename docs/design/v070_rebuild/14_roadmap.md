# 14: roadmap, the single map of the programme in the owner's order

| Field | Value |
|---|---|
| Status | roadmap, revision 3, 2026-11-16. State as of the ledger entry of 2026-11-16 00:44 and `int/native` at `e28fe4ea6`; stage 2 (section 3) as of `int/native` at `631abb042` (ledger entry of 2026-11-16 03:28) |
| Revision note | revision 3 follows the owner's binding order of 2026-11-15 (engine, backend in Rust, gallery frontend, database and clustering, cloud). Revision 2's track table, gap list and phase names are replaced; every gap of revision 2 is carried into section 9. Git holds revision 2 |
| What this file is | what is done, what is being built, what comes next and why, stage by stage. Task detail lives in the plans. Where this file and a plan disagree on order, this file wins; on task content, the plan wins |
| Branches | `int/native`: the Rust integration branch (engine and server share the workspace `native/`). `int/b4-stateless`: the Python product, pushed after green sweeps, the reference and parity partner until each family moves to Rust; fixes only. `int/storage`: storage step 1, frozen as the oracle the Rust fold must pass, and the storage measurements. `lane/plan16`: plan 16, waiting for its rewrite |

## 1. The person's job and the objective

A person points OpenProcessor at their photos or uploads them. The detector finds the objects. The
person labels their own classes cluster by cluster, then trains a model on them. Around that, their
photos become a library they can search, organise and enjoy (timeline, places, people, events)
([VISION_AND_GOALS.md](../../VISION_AND_GOALS.md), "The user's job").

The objective, in the owner's words: "process this image data set as fast as possible and
efficiently and make it work for other tasks as well for images". The measure is photos per second
per GPU and per CPU core from "photo in" to "stored, findable and on the map", and the latency of
that path for one photo. The local app stays the product; a cloud version is built toward from now,
so nothing is written twice.

Standing rules (owner, 2026-11-15): no authentication is built now, and a public deployment needs it
first; no 0.7.0 tag until the whole programme below is done; no upgrade path from 0.6.0 (wipe and
re-ingest); no new Python product code (fixes only, said to the owner first); nothing is written that
a later step rewrites; selective checks per slice, one full sweep per large merge.

Already built before this order, on `int/b4-stateless` (pushed at `e5f88dc58` after the green full
sweep 1115i, 2026-11-15): PostgreSQL for control data and jobs, one Ultralytics version, the
stateless API, the Cropwright labelling workflow (plans 12 to 12f), ingest on the frame engine with
its reliability fixes ([13a](13a_ingest_on_engine_plan.md)), the crop encoder as its own model at 224
([13b](13b_crop_vectors_plan.md)). It is the product the Rust work replaces family by family.

## 2. Stage 1: the Rust image engine (plan 17)

**What the person gets.** Each photo is decoded once on the GPU and every model input is made in one
Rust process, so the card, not the Python interpreter, sets the rate. Plan:
[17_native_engine_plan.md](17_native_engine_plan.md).

**Done.**

| Item | Evidence |
|---|---|
| The Rust workspace and its gates (format, lint, docs, tests, licences, CI job) | on `int/native` since the first slice; all Rust gates green at `519c0e365` and at each merge after |
| The first slice (N0): GPU core `gpu_frame`, the front of protocol version 1, plan parser, detector, frame vector, crop vectors, load generator, image, test bed | code-complete at `e5f0e95d2`; standards pass (complexity, duplication, docs) at `1a3311290`. Shakedowns 1 to 3 (6 photos per set, loaded host, indications only): bit-identical with the Python engine at 1 photo in flight on COCO and set P (a private set of real-world 20 MP vehicle photos), both plans; crop vectors 46 of 46 bit-identical at 1 in flight; with stand-in models at 32 in flight 1,941 photos/s against Python's 170 |
| The parity rule above 1 photo in flight | decided 2026-11-15 from shakedown 3: the Python engine differs from itself by 2.5e-4 in crop-vector cosine when only the batch composition changes, the same size as Rust against Python; written into 17 section 1.2 |
| Every branch of N1: faces (N1b), text reading and thumbnails (N1c), the non-JPEG route, formats, statuses and body transport, stored boxes, late rounds, colour and the crop cap (N1a) | merged into `int/native` at `84c1dde74` (faces `999b69127`, route `519c0e365`, boxes and colour `916edcbf4` and `e00387bc5`, text and thumbnails `84c1dde74`); 226 library tests and goldens green, Python native tests 118 passed. **Merged, not yet run on a GPU** |
| The engine's own plan JSON reader, so the server crates share one serde_json | merged at `e28fe4ea6`: 3,000,000 differential inputs, 0 differences |

**In progress (2026-11-16 00:44).** Rebuild of the binaries and the native image from `int/native`;
then the parity shakedown and the GPU checks of the N1 combination.

**Next, in order.**

1. GPU checks of every N1 branch and of their combination (the hold list in the ledger: colour with
   text and thumbnails on both routes, body transport with text and faces, budget refusals on the
   non-JPEG route, stored boxes with text and thumbnails, the nvJPEG byte metrics) (17 N1 gates).
2. The trimmed engine comparison: the full parity hold on the standard lists with the floor pair
   (Python at 1 against Python at N in flight), plus headline rate and engine-capacity rows (both
   plans, 32 in flight, `coco2k` and `setp2k`, one run) on a quiet host (17 bars G1 to G9; the full
   table is measured once at N3).
3. Packaging, profiles, faults and the sweep (17 N2): Trivy, the lane count per GPU profile from
   the measured nvJPEG bytes per megapixel, the fault rows, the two-card row, and reading a photo by
   key from the landing zone (the item plan 18 section 12 gives the engine).
4. The switch and the deletion (17 N3): one engine, the Python `frame_engine/` deleted, the final
   table under the evidence clause (the paper's main table).
5. After N3: re-plan the card-budget levers (both cards, model build and batching settings) against
   the native engine (plan 13 items P2, P5, P6; gap G9).

**Owed before.** The owner's answer on the eight refused formats before N3 (section 5). The nvJPEG
bytes per megapixel before N2's profiles. The face benchmark harness through the engine before N1b's
gate counts as met.

**Exit.** 17 N3's exit: one engine in the tree and the stack, a green sweep, the final table of plans
F, C and A. Moving `gpu_frame` into the owner's Rust serving project happens when the owner
publishes that project.

## 3. Stage 2: the API and the Python backend replaced by Rust (plan 18)

**What the person gets.** One robust upload endpoint for one or many files, the same one Cropwright
uses today; photos stored durably (local disk here, S3 in the cloud) and never marked stored before
they are; ingest at the card's rate; Rust as the API, the database writer, the image processing, the
housekeeping and admin work, the job queue (on PostgreSQL, replacing Celery) and the clustering where
a measurement allows. Plans: [18](18_backend_rust_port_plan.md) (decisions, order, storage and
upload path, cloud benchmark), [18a](18a_backend_rust_port_slices.md) (workspace, tests and parity,
the slices that start now), [18b](18b_backend_rust_port_later_slices.md) (later slices, the mapping
of the storage slices, traceability).

**Done (2026-11-16, `int/native` at `631abb042`).** Plan 18 revision 1 with its 13 decisions and five
answers, and decisions 14 to 20 of 2026-11-16 (contract-first, one owner per landing file, one TLS
stack, the derived-file cache, the API capacity rows closed). Merged and tested: workspace
preparation (R0w); the storage layer (R1s: local and S3-compatible, never half written, with the
cache); capture reading and content hashing (R1x); the job runner core, proven on PostgreSQL 18
(R1j, with TLS to the database); the engine client (R1e); the upload receiver up to the landing zone
(R1u, not yet mounted); the upload load generator (R0g); the core contract baseline (R0c: 118
operations); the differential runner, parity ledger and pending list (R0d). The contract spike (R0s,
measurement M15) is done and was not merged; its result is contract-first. The fold oracle export
(R0x) is done on its own branch for R1b. Not merged: seeding over HTTP (R0e), which waits for the
owner's answer (section 8).

**In progress.** The server front (R1a), in its fix pass after an independent review; it is the one
stage 0 lane not yet merged. Plans 18, 18a and 18b were brought in step with the above on this date.

**Next, in order** (18 section 3.1; 18a section 5; 18b sections 1 and 4).

1. One sweep of the merged tree once R1a is merged: the database-tier tests, the S3-compatible
   server test of the storage layer, the server image (with ExifTool, Perl and `setpriv`), the live
   and browser tiers through the front, the differential dry run and same-stack read run, the upload
   load generator's loopback row, and the full pre-commit and pre-push hooks (18a section 5 lists what
   each slice owes). The rest of stage 0 after that: seeding over HTTP (R0e) once the owner answers.
2. Stage A: the record slice R1b after measurement round 4's first part (the part needed before R1b,
   plan 15b section 15); periodic ticks in Rust and the Python claim loop, Celery leaves (R3a, which
   also decides the claim for a one-kind runner behind a large backlog); live events from Rust,
   Valkey leaves (R3b); projects on the record (R11a); **Rust ingest switches on**, uploads and
   folders, S3 storage (R1c, after N3); inference routes on the engine (R2a, after N3).
3. Stage B: gallery reads (R4); labelling acts, review and counts (R5); erasure (R5e);
   configuration, settings and models (R11).
4. Stage D, after stage 4's slices: the Python API removed and the schema out of Alembic (R12). This
   closes the owner's step 2.

**Owed before.** R1c: real objects per photo (M4), ExifTool's own cost (M5), today's upload route as
the baseline (M10), storage write cost and SHA-256 (M18); the Rust capture reader (M16) is optional.
M1 (API CPU profile) and M2 (30-process and 1-process grids) are not run: the capacity grid closed at
27 of 33 rows and the missing rows measure the Python API the Rust server replaces, so the process
count stays at the compose default. R3b: `NOTIFY` at ingest commit rate (M17). R2a: core route
profile (M6), and a decided response shape for each core route the contract leaves open (R2a, R2b and
R11 name them). The priority of reading by key: the path-mode spike (M3).

**Exit.** R1c's go bar (API host CPU per photo at most 0.5x Python's, 18b R1c) with stage A green;
the step closes at R12 with the pending list empty and every parity ledger closed.

**Windows.** Between R1c and each family's slice, families not yet moved show nothing for photos
ingested after the switch; the full sweep keeps proving them against the Python reference
(18 section 3.3).

## 4. Stage 3: the frontend gallery update

**What the person gets.** Their photos as a library in Cropwright: timeline, albums and smart albums,
places and the map, people, curation (exclusions, stacks, culling), events, trips and memories.
Plans: [the photo-library plan](../photo_library/00_photo_library_plan.md) and its phase files
P2a-0 metadata, P2a-1 organise, P2a-2 people, P2b curate, P2c create.

**Done.** The photo-library plans (2026-11-13) and the owner's eight answers
([90](../photo_library/90_owner_questions.md)); the Cropwright refinement plan 12f built. No gallery
feature is built.

**In progress.** Nothing.

**Next, in order.**

1. Re-base the photo-library plans on the record and the Rust backend (gap G3): no search-index
   roles, no Celery, no Python routes; the gallery's read shapes listed and measured like gate A,
   two- and three-hop questions included (G20); one owner for face grouping (G7); the owner's calls
   on relations between people (G18) and stored scene tags (G19).
2. The gallery frontend update starts when R4 (gallery reads) merges (18b R4); then P2a-0, P2a-1,
   P2a-2, P2b, P2c, each after the previous phase's exit bar and the owner's walk. The metadata
   reader of P2a-0 is agreed with plan 18's decision D2 (does ExifTool stay on the hot path).
3. The engine needs of the gallery without an owner (G4): HEIC and RAW decode, quality signals,
   per-face signals; moved and deleted source files.

**Exit.** Each phase's exit bar of the photo-library plan, walked by the owner; the frontend checks
and browser tier green; pages viewed at desktop and 430 px.

## 5. Stage 4: the database and the clustering features

**What the person gets.** One decision labels a whole group; search, near-duplicates, regions,
import, export and training on the record; one database when the measurements allow it.

**Done.** The storage design (plans [15](15_storage_plan.md), [15a](15a_storage_schema.md); 15b and
15c on `int/storage`): kept as design and SQL, their Python slices superseded by plan 18 (18b section
6). Storage step 1 on `int/storage`, frozen as the oracle (45,000 differential sequences). Gate A at
1 and 10 million rows: deep list page 1.3 / 2 ms (p50 / p99, 8 clients) in PostgreSQL against 32.3 /
52 in the search engine (15b section 1.1). Gate B round 1 at 957,000 real vectors: recall at 10 0.997
for pgvector at a 0.26% filter, p50 2.8 ms (15 section 1). Cluster partitions and placement of new
photos built in the Python product.

**In progress (2026-11-16 00:44).** The labelling measurement (plan 16's M16-0) and storage
measurement round 3 (15b M1 to M10), both running. M16-0 finished later on 2026-11-16 (plan 16
revision 1 rests on it).

**Next, in order.**

1. Plan 16 (the cluster, label and train loop), revision 2, for the Rust backend:
   [16](16_label_train_loop_plan.md), [16a](16a_label_train_loop_slices.md),
   [16b](16b_label_train_loop_schema.md); evidence in
   [labelling_decisions_m16_0.md](../../research/labelling_decisions_m16_0.md) (gaps G1, G6).
   Part A moves today's loop to the record and Rust with the same behaviour for the person (18b R8
   split into R8-1 and R8-2, the label-act precondition added to R5); its numeric library (L16-1)
   can start now and M12 is its only entry measurement. Part B (a random check sample, a group-count
   rule, label scopes, a classifier on the person's labels, growth without constant refits) waits
   for M16-1, M16-2, M16-3, M16-6 and M16-7, each with its bar.
2. Storage round 3 written up; its numbers are the entries of R1b, R4, R5 and R6b.
3. Gate B round 2 at 10 million real crop vectors, with one shared build of 100,000 and 1,000,000
   crop vectors for M12, M13 and M14 (18 section 8).
4. The slices of 18b stage C: vectors and search (R6a), near-duplicates and the holdout (R6b),
   regions (R7), clustering and the cluster screens (R8), import, combine, export and training inputs
   (R9), machine helpers and re-runs (R10), core data on the record and OpenSearch leaves if gates A
   and B say so (R2b, decision D4). Then the acceptance walk on the Rust stack, then R12 (section 3).

**Owed before.** R6a: gate B round 2. R6b and R7: 15b M9. R8: k-means at scale in Rust against faiss
(M12), 15b M10. R6b, R10, R2b: exact all-pairs top-k (M13). R2b: gates A and B and D4. The owner's
call on the 2-D map method after M14.

**Exit.** 15c section 4's acceptance walk once on the Rust stack (re-ingest, derived-store rebuilds,
the scripted labelling walk, two API processes and two workers under load, gate A and B bars).

## 6. Stage 5: the cloud-like deployment

**What the person gets.** "A cloud version of our app for people to store and organize their photos
for a fee, or for small businesses to run", from the same images as the local app.

**Done.** The benchmark specification (18 section 5); the deploy shape the port changes and the
eleven multi-user seams as non-goals (18 sections 6, 6.1); the local compose stack, installer and
GPU profiles.

**In progress.** Nothing: the upload load generator (R0g, stage 2) is merged; its 600 MB/s loopback row
and M10 are owed to the sweep.

**Next, in order.** A deploy plan (the rest of G14: compose overlay for S3 and a remote landing host,
managed PostgreSQL with pgvector, TLS, backups, secrets); the benchmark runs at 2,000, 100,000 and
1,000,000 photos; the local rows on one and two A6000s.

**Owed before.** B1 to B3 need R1c; B4 needs R4; NAS read and write rates (M11) before the first
scale run; a test of hardware JPEG decode on the L4. Authentication before any deployment reachable
from the internet.

**Exit.** B1 to B6 recorded at the three sizes; the owner sets the bars after the first baseline.

**The benchmark in five lines** (18 section 5). Photos arrive through the upload route as the
frontend sends them, from a compiled load generator. B1 photos per second per GPU, B2 per CPU core,
B3 time to "stored and findable" (p50, p99), B4 on the map and in a time group, B5 cost per million
photos, B6 the same on this machine. Sizes 2,000 (regression), 100,000 (headline), 1,000,000
(scale), half set P and half public photos. `g6.4xlarge` is the named example host; the host is a
parameter.

## 7. What stays Python, and why (18 section 2.3)

| Stays | Why |
|---|---|
| `probe`, `train`, `model_export`, `bakeoff`, `promote` | Ultralytics, PyTorch, TensorRT and ModelOpt, ONNX Runtime and `onnx` have no Rust equivalent |
| `embedding_viz` (the 2-D map, UMAP) | no maintained exact Rust port; M14 measures map quality so the owner can decide |
| The segmenter and vLLM servers | their own images |
| `cluster_run` | Rust, CPU or GPU as M12 decides; no Python fallback (plan 16a) |
| `item_scores`, `near_dup_score`, `near_dup_core_group`, `persons_cluster` | Rust; M13 on real crop vectors decides CPU or GPU inside Rust. No Python fallback: a kind that misses its bar on both goes to the owner as a question (plan 18 section 2.3) |

Everything else is Rust. A Python job reads and writes only the job tables and its own result tables
and never folds a person's decision; it takes work through a claim loop on the same `jobs` table.

## 8. Measurements and decisions owed

| Item | Decides | Blocks |
|---|---|---|
| Eight formats the Rust engine refuses (answers `decoder`) where the Python path opened them: PPM, TGA, ICO, PCX, QOI, JPEG 2000, SGI, palette TIFF | owner | N3 (the accepted set must equal today's unless the owner drops them) |
| Seeding over HTTP (R0e): Python or Rust tooling (the core contract baseline R0c and the differential runner R0d are merged as Python tooling) | owner | R0e, so R1c's parity |
| NAS backup of two private map files | owner | no stage; a data-safety step |
| A faster content hash, only if M18 breaks the B1 budget at 20 MP | owner | R1c |
| The 2-D map method after M14 | owner | `embedding_viz` in R10 |
| Relations between people (G18), stored scene tags (G19) | owner | the gallery re-base |
| Authentication | owner, built later | any internet-reachable deployment |
| The Rust engine counts `op_frame_engine_colour` on the non-JPEG route, the Python engine does not | coordinator | N3's metric diff |
| Three Python face behaviours the port matched that look wrong (NaN landmarks fail the photo; collinear landmarks embedded; recogniser row count unchecked) | coordinator | N3 |
| GPU checks of N1; the trimmed engine comparison | measurement | N2, N3 |
| Face benchmarks (LFW, CALFW, CPLFW, WIDER FACE) through the engine: no harness yet | measurement | N1b's gate |
| nvJPEG device bytes per megapixel | measurement | N2's lane counts |
| Two-card row | measurement | the card budget |
| Hardware JPEG decode on the L4 | measurement | the cloud runs |
| M4, M5, M10, M18 (objects per photo, ExifTool, upload baseline, storage write cost); M1 and M2 are not run (18 section 8) | measurement | R1c; M5 and M16 settle D2 |
| M3 path-mode spike | measurement | priority of reading by key |
| M15 contract spike (R0s) | measurement, done 2026-11-16: D6 is contract-first | nothing |
| M17 `NOTIFY` at commit rate | measurement | R3b (Valkey leaves) |
| M6 core routes; M7 read routes; M8 static facts; M9 region worker | measurement | R2a; order of stages B and C; R1a; R7 |
| M11 NAS rates | measurement | the first scale run |
| M12 k-means at scale; M13 exact top-k | measurement | R8; R6b, R10, R2b (decision D5) |
| The labelling measurement (plan 16's M16-0) | measurement, done 2026-11-16 | plan 16 revision 2 |
| Plan 16's M16-1 (the owner reviews 60 pages), M16-2, M16-3, M16-6, M16-7 | measurement (M16-1 with the owner) | plan 16's Part B items, each behind its bar; Part A needs only M12 |
| Storage round 3 (15b M1 to M10) | measurement | R1b, R4, R5, R6b, R8 entries |
| Gate B round 2 at 10 million vectors | measurement | R6a |
| Gates A and B together | measurement | D4, so R2b and whether OpenSearch leaves |

## 9. Revision 2's gaps, carried forward

**Closed.**
- G5 core routes on the engine and the record: plan 18, slices R2a and R2b.
- G8 plan 13's Rust engine text: closed in revision 2 (plan 13 items marked superseded by plan 17).
- G10 plan 17's scope line and the `gpu_frame` move: closed in revision 2 (17 N3 exit).
- G16 stale status text: closed in revision 2; the vision's release section is stale again (it says
  "the API where measured" and Celery and Valkey as foundation), owner of that file to refresh.
- G17 the ingest reliability rerun row: closed in revision 2.

**Moved into a stage and item.**
- G1 the cluster rule: stage 4 item 1 (labelling measurement, then plan 16 rewritten; R8 builds it).
  The 13b sentence about clusters within a detector class is still to be corrected by that rewrite.
- G2 clustering fit time at scale: plan 18 M12, before R8 (stage 4).
- G6 the labelling and training loop: plan 16, rewrite after the labelling measurement (stage 4).
- G9 the card-budget levers: stage 1 item 5, after N3 (owner: plan 13).
- G13 frame-target search with two homes: R6a (stage 4); the slice brief names one owner.
- G14 deploy plan: the parts the port changes are in 18 section 6; the rest is stage 5's deploy plan.
- G15 plan 15 says the vector step need not wait for gate B: superseded, R6a's entry is gate B
  recorded (18b); plan 15's sentence still reads the old way (owner: plan 15).

**Still open, with their owner.**
- G3 gallery plans on the search index: stage 3 item 1 (the photo-library plans).
- G4 gallery engine needs (HEIC and RAW, quality signals, per-face signals, moved files): stage 3
  item 3; a plan 17 revision after N3 for the engine parts, R1c's source jobs for moved files.
- G7 face grouping with two owners: the photo-library people plan, in the re-base; the job is
  `persons_cluster` in R2b (stage 4).
- G11 plan 10 (v0.8.0) predates the record and the Rust decision: re-based after this programme.
- G12 video has two homes: planned once, after the gallery (section 10).
- G18 relations between people, G19 scene tags: owner decisions in the gallery re-base.
- G20 multi-hop gallery reads: stage 3 item 1, measured like gate A.

## 10. Deliberately not built now

| Not now | Where it attaches later |
|---|---|
| Authentication and authorisation | a tower layer in the Rust API before the extractors (18 section 6.1 row 11); the entry condition of any public deployment; plan 10 F5 |
| Multi-user seams (accounts above projects, per-account registries, quotas, per-account queues) | the eleven places in 18 section 6.1 |
| Video (P3, P4) | the photo-library video outline ([40](../photo_library/40_p3_p4_video_outline.md)), planned once after P2c; plan 10 F7 folds into it |
| The desktop app | the same wire contract ([50](../photo_library/50_desktop_direction.md)); reviewed after P2c |
| Orchestration, autoscaling, the GPU arbiter overlay | the later deploy plan (18 section 6) |

## 11. How the work runs

- The coordinator writes and decides the plans, reads the risky diffs, judges every measurement and
  runs every merge, GPU hold and sweep. A plan is a file in the repository before a lane starts.
- Lanes: one agent per lane in its own worktree and branch, one commit per concern, hooks on, never
  `--no-verify`, never merge or push. Light checks only; database, compose, live, browser and GPU
  runs are owed to the wave. No two lanes write the same code: shared shapes are fixed in writing
  first.
- One full sweep per large merge, on the merged tree; reds go back to one fix lane. A push of an
  integration branch follows a green sweep; never `main`, a tag or a release without asking.
- GPU runs name the card after checking peer sessions, and set both `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  and `CUDA_VISIBLE_DEVICES`. GPU 1 belongs to another project.
- Every speed claim follows 17 section 5's evidence clause (public sets and set P as numbers, the
  hardware with its CPU, versions, the command, the bound). This host's CPU is the baseline; nothing
  is tuned to it.
- A result that misses its bar is reported as a miss with its number; a bar moves only in writing.
- In the owner's Rust serving project: a worktree and branch per lane, no push without the owner's yes.
