# 14: roadmap, the single map of all remaining backend work and what follows it

| Field | Value |
|---|---|
| Status | roadmap, revision 2, 2026-11-15. Rewritten after the owner's decisions of that day (the engine in Rust now, the whole backend to Rust by measurement, PostgreSQL as the record). Every state below was checked against the code and the branches on that date, not taken from a plan's own status line |
| What this file is | the one place that names every track, its detailed plan, its state, what it waits for, what comes next and in which order. Task detail lives in the plans; where this file and a plan disagree on order, this file wins; on task content, the plan wins |
| Reads with | [00_master_plan.md](00_master_plan.md) (the 0.7.0 waves, done up to W4 and the workflow rebuild), [13](13_backend_speed_plan.md), [13a](13a_ingest_on_engine_plan.md), [13b](13b_crop_vectors_plan.md), [15](15_storage_plan.md), [15a](15a_storage_schema.md), [17](17_native_engine_plan.md); plans 15b (reads) and 15c (slices) are on branch `int/storage` only; the gallery plans in [../photo_library/](../photo_library/00_photo_library_plan.md) |
| Branches | `int/b4-stateless` (the integration branch, pushed after each green sweep); `int/native` (plan 17, merges into the integration branch when N0 is a go); `int/storage` (plan 15, merges back when plan 15 section 11 passes) |
| Old numbers | revision 1's phases and sections are cited by 13b, 15 and 17. Read them as: phase 1 = track "ingest on the engine"; phase 1b = "crop vectors"; phase 2 = the storage tracks; phase 3 = "core routes on the engine and the record"; phase 4 = "labelling and training loop"; phase 5 = the gallery tracks; phase 6 = "deploy"; phase 7 = "API port to Rust"; section 3 = section 6 here; section 4 = section 4 here; section 6 = section 8 here |

## 1. The goal, and how the tracks serve it

The user's job comes first ([VISION_AND_GOALS.md](../../VISION_AND_GOALS.md), "The user's job"): a
person adds photos; a detector finds the objects; the person groups them by cluster, labels them in
their own classes (with SAM-like box help and an optional VLM), and trains a custom detector; the
same photos are also a first-in-class gallery (timeline, places, people, curation, memories).

The goal, in the owner's words condensed: the best NVIDIA image-processing pipeline, a
first-in-class photo gallery, and projects for cluster, label and train, all in one app; production
quality; deployable here, on other machines and in the cloud; the backend moved from Python to
Rust. The value is the outcomes (trained models, an organised library, polished memories), built in
measured phases: **image engine, then image organise, curate and create, then video processing,
then video memories.**

How the tracks serve it: the engine tracks make "photo in" to "stored and findable" fast (one
decode, every model from one frame, a Rust process). The storage tracks make every person's
decision safe and every screen's list, count and search fast at 10 million objects. Clustering,
labelling and training turn stored objects into a trained model. The gallery tracks build on the
same record. The API port, deploy and release make it one product that runs anywhere.

## 2. The map

State words: **built and merged** (on `int/b4-stateless`), **built on a branch**, **measured**,
**planned**, **not planned**. Sizes: small, medium, large, with lanes or slices where a plan counts them.

| Track | What it delivers to the user | Detailed plan | State today | Waits for | Next step | Size |
|---|---|---|---|---|---|---|
| Native frame engine (N0 to N3) | each photo decoded once on the GPU and every model input made in one Rust process, so the card, not the interpreter, sets the rate | [17](17_native_engine_plan.md); code on `int/native` | built on a branch, N0 in part: Rust workspace and gates, `gpu_frame` (GPU self-test passed on GPU 2), the front, plan parser, geometry, stub pipeline, contract test run against both engines. Lanes N0-d (real pipeline) and N0-e (load generator, bench scripts, image) coding | N0-d and N0-e; the coordinator's holds for bars G1 to G9 (17 section 0.2) | N0 go on plans F and C, merge `int/native`, then N1 | large: N0 5 lanes, N1 3, N2 1, N3 1 |
| The card budget | more photos per second per card once the engine is native: both cards, model build and batching settings | [17](17_native_engine_plan.md) section 0a; items P2, P5, P6 and section 8 of [13](13_backend_speed_plan.md); the two-card row of 17 N2 | measured on paper (17 section 0a); the cheaper-crop-encoder lever is closed (13b section 8); nothing built | N3 (a gain shows only through the native engine); N2's two-card row | re-plan P2, P5, P6 against the native engine (gap G9) | medium |
| Ingest on the engine, reliability | a photo stored whole or not at all, one photo at a time from the engine to its write, no API decode | [13a](13a_ingest_on_engine_plan.md) | built and merged: lanes L0, L1, L2, L3, L4, L7 and the engine-call reliability fixes. Measured (13 section 10, rows "D"): set P 1.17x the old code, COCO 0.98x (inside the spread, encoder-bound); rows with failed photos are invalid and owed again. L5 and L6 not started | the rerun of the invalid rows in the next sweep | record the rerun in 13 section 10; close 13a at N3 | small (rows only) |
| Crop vectors | crop vectors that tell objects of one detector class apart at every object size | [13b](13b_crop_vectors_plan.md) | built and merged: the crop encoder is its own model at 224 with letterbox, 336 a deployment choice; no smaller encoder replaces it (measured) | nothing | the crop-space reds of sweep 1115f (fix lane running) | done |
| Storage: the record | a person's decision is never lost; every machine result can be rebuilt | [15](15_storage_plan.md), [15a](15a_storage_schema.md) | step 1 built on `int/storage` (migration 0010, repositories, the fold `apply()`, the erasure catalogue, the vector schema and role); its database tests green on a real PostgreSQL | nothing | wave A | part of the slices |
| Storage: reads and indexes ("the database queries") | every list, count, facet and page answered by SQL within the page bars at 10 million objects; a write visible to the next read | 15b on `int/storage`: the 80 read shapes, indexes X1 to X32, the capped-count rule, sorts, the trigram index, box rows, job readers | gate A measured at 1 and 10 million rows; measurement round 3 (items M1 to M10) running | round 3 | the entry conditions of each slice | part of the slices |
| Storage: summaries | class counts, statistics and cluster cards without scanning, exact at every snapshot | 15b section 6 (`int/storage`): insert-only delta log, one emitting function, a compactor, rebuild without a lock | planned | M1 to M4 | slice S1a-i | part of the slices |
| Storage: slices and waves | the cut-over feature by feature, the old path deleted in the same slice | 15c on `int/storage` | planned: step 1 plus 16 slices in waves A to G, each with entry conditions | round 3; the owner's answer on the 0.6.0 upgrade code before wave C; gate B before S3 and S5 | wave A: S1a-i, S1b-i, S1b-ii | large: 16 slices, 7 waves |
| Vectors and the vector index (gate B) | search that stays fast with filters at 10 million vectors and more | [15](15_storage_plan.md) R6 and section 7; [15a](15a_storage_schema.md) section 6 | decided: every vector is kept in PostgreSQL, one table per space, `halfvec` (half precision) stored PLAIN, beside its model version; the approximate index is derived and its default is gate B's. Measured, round 1 (957,000 vectors, 0.26% filter): recall at 10 0.997 for pgvector and Milvus, 0.996 VectorChord, 0.860 a FAISS graph; pgvector p50 2.8 ms. Round 2 not started; it must confirm, at 10 million or more real crop vectors, pgvector (HNSW on half precision; 1-bit codes with a rescore) against VectorChord, Milvus (graph, disk, GPU), Qdrant and OpenSearch on build, size, recall unfiltered and at 3% and 0.3% filters through the real join, p50 and p99 at 1, 8, 32 clients, inserts while searching, filter-field updates, restart, licence; a second service must beat the in-record one by more than the spread on a bar a user feels. Its precondition, the chosen crop encoder, is now met | a coordinator hold; 10 million crop vectors from the 224 encoder on public sets | run round 2 | medium (one measurement) |
| Leaving OpenSearch | one database to run, back up and port | section 2.3 | still the store of every curation screen, the core routes and the vector searches today | the storage slices, gate A's outcome, gate B, the core-routes track | see 2.3 | rides the tracks |
| Clustering | one decision labels a whole group | no single plan: section 2.1 says which document owns which piece | partition foundation built and merged; the decided cluster rule measured, not built | see 2.1 | see 2.1 | medium |
| Near-duplicates | duplicates found, grouped and held out together; the gallery's stacks | [09d](09d_research_near_duplicates.md); slice S5 (15c); [20 curate](../photo_library/20_p2b_curate.md) section 3 | built and merged (runs, exact key, holdout guard); research experiments after the tag | S2a, S3, gate B (S5's entry) | S5 in wave D | medium (1 slice) |
| Search | find photos and objects by text or by picture | [09a](09a_research_embeddings_search.md); slice S3 (15c); item S1 of 13; feature F2 of [10](10_v080_platform_breadth.md) (one row per photo, frame target) | built and merged on the search index (text tower on TensorRT); sweep 1115f found a live search-by-picture test under its bar (being checked) | gate B; S1a-ii, S1b-i | S3 in wave C | medium |
| Labelling and training loop | review, auto-label, the assist loop, SAM-like box proposals, train, promote | [12](12_review_workflow_audit.md) to [12e](12e_workflow_execution_order.md), [09e](09e_research_auto_label_assist.md), tracks T4, T5, T6, T10 of 13; storage slices S2a, S4, S7, S8a to S8c | built and merged (the workflow rebuild and its walk); each act's storage moves in its slice; T10's few-labels head measured, not built; T5 (segmenter cost) not measured; no plan for the loop on the new store (gap G6) | storage waves B to F | write plan 16 after wave B | large |
| Core routes on the engine and the record | /detect, /faces, /analyze and /search on one engine and one store; the 12 GB card profile | items A1, E3, E4 of 13; 17 N1 and N2 (branches, profiles) | engine branches built in Python, being ported in N1; the API side not planned; the core data stays on the global search indexes until this track (15c, answer 7) | N3; storage S9 | a plan file (gap G5) | medium to large |
| API port to Rust | an API that keeps up with the native engine; one backend language | none; the order and the bar are section 4 | not planned | N3 and slice S1a-ii | the ingest-path measurement of section 4 | large |
| Frontend (Cropwright) | screens that follow served data | [12f](12f_ux_refinement_plan.md); the frontend rows inside each storage slice (15c section 2, 15b section 5) | 12f built and merged; storage's frontend changes ride their slices | the slices | none on its own; see section 3 (what must not start) | rides the slices |
| Gallery P2a-0: metadata | one ExifTool reader for photos and videos, GPS tracks, keywords, captions, edits, XMP sidecars | [05](../photo_library/05_p2a0_metadata.md) | planned (2026-11-13) on the search index; must be re-based on the record (gap G3). The ExifTool pool and the capture reader of today are built and merged | the re-base; slice S1b-ii | the re-base pass | medium: 6 lanes |
| Gallery P2a-1: organise | timeline, albums, smart albums, places | [10](../photo_library/10_p2a1_organise.md) | planned, same re-base | P2a-0 exit; P1 needs (2.2) | after P2a-0 | large: 8 lanes |
| Gallery P2a-2: people | people from faces, confirm, merge, split, forget | [15](../photo_library/15_p2a2_people.md) | planned, same re-base | P2a-1 exit; faces in ingest; face grouping (2.1) | after P2a-1 | medium: 3 lanes |
| Gallery P2b: curate | exclusions, stacks, assisted culling | [20](../photo_library/20_p2b_curate.md) | planned, same re-base | P2a-2 exit | after P2a-2 | large: 7 lanes |
| Gallery P2c: create | events, trips, calendar, memories, slideshow videos | [30](../photo_library/30_p2c_create.md) | planned, same re-base | P2b exit | after P2b | large: 8 lanes |
| Video P3 and P4 | videos as library items; video memories | [40](../photo_library/40_p3_p4_video_outline.md) (outline); feature F7 of [10](10_v080_platform_breadth.md) | outline only | P2c exit | plan after P2c (one owner: gap G12) | large |
| Deploy | the same images here, on other machines and in a cloud account | section 7; 17 N2 (profiles); 15 section 8 and slice F (files); features F5 and F10 of [10](10_v080_platform_breadth.md) | the compose stack, installer and profiles built and merged; files layer planned; authentication not built | slice F; N2; F5 | a deploy plan (gap G14) | medium |
| Desktop direction | a standalone app on the same contract | [50](../photo_library/50_desktop_direction.md) | direction only | P2c | review after P2c | not planned |
| Release | acceptance, docs, the paper, publishing the engine | [07](07_release_acceptance.md); 17 section 5 (evidence) and N3 (the final table); [WHITEPAPER.md](../../WHITEPAPER.md) | 07 not started; which tracks go into 0.7.0 is open (section 9) | the owner | the release cut | medium |

### 2.1 Clustering, piece by piece

| Piece | Decided | Owner document | State | Missing |
|---|---|---|---|---|
| Method and cluster count | keep k-means; partition by detector class, then K about n_c // 80 (track T7: 0.707 labelling decisions per item against 0.971 for the shipped rule) | 13 section 10, row T7 | measured; the code still fits K = min(512, n // 6) over the whole pool (`src/services/clustering/run.py:42-43`) | **no plan or lane builds it** (gap G1) |
| New photos | assigned to the serving partition, never refit with carry-over (T7: a refit moves 24 to 46% of earlier items) | 13 T7; 13a L4; slice S6 | placement at ingest built; after the cut-over it is membership rows written in S1a-ii's transaction | nothing |
| Where it is computed | a queued job, CPU fit, reading every vector of its pool | `src/services/clustering/job.py`; 15c S3 and S6 | built; vectors read from the search index today, from the record's vector tables after S3 | fit time at 10,000 to 1 million vectors not measured (gap G2) |
| Where it is stored | versioned partitions, stable identities, memberships in PostgreSQL; readers join the serving partition; activation flips a pointer | 15 section 13; 15c S6 | partitions built and merged, still projected into every item document; the side table replaces that in S6 | nothing |
| Cluster screens at scale | cards and facets from partition-keyed summaries filled before the pointer flips | 15b sections 6.7 and 13 (M10) | planned | M10 |
| Few-labels head for fine-grained classes | logistic regression on stored crop vectors (T10) | 13 T10 | measured, not built | an owner: plan 16 (gap G6) |
| Face grouping | average linkage resumed from existing groups, a review band, same-photo cannot-link (F3) | 13 item F3 **and** photo-library 15 lane L-P1 | measured; the code still runs union-find (`src/services/face_identity/person_partition.py:66`) | one owner (gap G7) |
| GPU fit | adopted only when a GPU method agrees with the CPU fit on real data and the fit dominates | [09b](09b_research_clustering.md) section 1; 13 T7 | not measured | waits for G2 |
| Research left | graph refinement, the open-set read, the visual judge and the blind human re-judge | [09b](09b_research_clustering.md), [09c](09c_research_clustering_protocol.md), [08](08_research_prototype_track.md) section 5 | after the tag | nothing |
| Events and trips (scene grouping) | the gallery's own grouping | [30](../photo_library/30_p2c_create.md) section 2 | planned | nothing |

### 2.2 What the gallery needs from the engine and from storage

The photo-library plan lists its needs from the engine as N1 to N10 ([00](../photo_library/00_photo_library_plan.md)
section 2.1, names provisional until the engine owner confirms them).

| Need | Engine | Storage | Owner gap |
|---|---|---|---|
| N1 faces in curation ingest; N7 per-face signals | `faces` and `face_embed` branches built in Python (13, F1), ported in 17 N1b; eyes-open and yaw are not computed | `faces`, `face_state` (15a section 2.5) | N7 has no engine owner (G4) |
| N3 thumbnails and placeholders | `derivatives` branch built, ported in 17 N1c | `photo_derivatives`; crops and thumbnails a cache by key (15 section 8) | none |
| N4 HEIC and RAW | not in 17 (N1a keeps today's accepted formats; HEIC is refused today; a spike measured two routes, 13 section 10) | nothing new | **no owner** (G4) |
| N5 whole-photo text | `ocr` branch built, ported in 17 N1c | `text_lines` | none |
| N6 quality signals (sharpness, exposure, aesthetic, utility class) | no plan computes them | `photo_signals` designed | **no owner** (G4) |
| N8 library-scale ingest, embedding only selected classes | 13 P6, the card budget | none | none |
| N9 frame-target search, one row per photo | none | slice S3 | two owners: 10 F2 and 09a (G13) |
| N10 moved and deleted source files | none | the files layer (15 section 8, slice F) | not assigned (G4) |
| Colour | the sRGB step built (13 section 10, row "Colour"), ported in 17 N1a | none | none |
| Capture metadata and GPS | none (P2a-0 owns the one reader) | `photo_capture`, `photo_state`; tracks and places stay as the gallery plans design them (15a section 2.5) | none |
| Capture formats beyond JPEG: Live Photos, edited renders and edit files, RAW+JPEG as one asset | HEIC and RAW decode: see N4 | `photo_files` (15a section 2.5) | bundles owned by [05](../photo_library/05_p2a0_metadata.md) section 2.5; decode is G4 |

### 2.3 OpenSearch: what still uses it, and when each use ends

Decided: PostgreSQL is the record and the only store for the curation screens (15 R1; 15b answers
every one of the 80 read shapes in SQL, gate A measured at 1 and 10 million rows). Today OpenSearch
still holds (574 files under `src/`, `scripts/` and the compose files name it on 2026-11-15):

| Use today | Where | Ends |
|---|---|---|
| The curation project indexes (images and items roles): every list, count, facet, review queue, cluster card, box row, near-duplicate scan and the vector searches | `src/services/curation/`, `src/routers/curation/`, `src/clients/curation_opensearch/`, `src/services/near_duplicates/`, the assist and auto-label services, the clustering projection, the periodic ticks (image counts, cluster refresh, open vocabulary), source backfills and location purge, training preflight, the workers under `scripts/curation/` | slice by slice in 15c; wave G (S9, revision `0018`) drops the project index records |
| The core API's global indexes: image ingest and search, faces and persons, text search, the core clusters and query routes, the face and text purges | `src/services/visual_search/`, `src/services/face_identity/`, `src/routers/search/`, `src/routers/faces.py`, `query.py`, `clusters.py`, `scripts/purge_*_data.py` | the core-routes track (G5); until then they stay by decision (15c, answer 7) |
| The 0.6.0 upgrade and rollback path | `src/services/upgrade/from_060/`, `scripts/db/upgrade_from_060.py`, `rollback_to_060.py` | when the owner confirms its deletion (question 1), else in S9 |
| Operations: the compose service and data directory, Dashboards (monitoring profile), heap sizing, install, health, metrics and error mapping, benchmark scripts | the compose files, `scripts/lib/`, `scripts/setup.sh`, `src/core/`, `scripts/bench/` | with the last product use |

OpenSearch leaves the stack entirely when all three hold: (1) wave G passes plan 15 section 11 with
gate A met in SQL, so no additive search copy (15b section 1.4) is needed for any shape; (2) gate B
picks an index inside PostgreSQL; (3) the core-routes track has moved the global indexes into
project tables. If gate A misses broadly or gate B picks OpenSearch, it stays as a derived store
fed from the record (15 R7), never as a record. The gallery plans still assume OpenSearch index roles
(a `faces` role, image-document fields, a projection drain): gap G3.

### 2.4 The gallery's knowledge graph, feature by feature

How the plans model it: relational tables, not a graph store. Entities (people, places, events,
trips, albums, keywords, memories) are rows; a photo's link to one is an edge row; a person's acts
are claims folded into state (15 section 4.5, 15a section 2.5). "No graph database: the questions
are two or three hops over a few entity kinds, which joins answer" (15 section 4.5).

| Feature | Owner (file, section) | State or gap |
|---|---|---|
| Moments and events (sub-events as chapters) | [30](../photo_library/30_p2c_create.md) section 2 (adaptive time gaps, place jumps), 4.4 | planned |
| Trips and home | 30 section 2 | planned |
| Places (reverse geocoding, drill-down, map, GPS tracks) | [10](../photo_library/10_p2a1_organise.md) section 3.4; [05](../photo_library/05_p2a0_metadata.md) section 2.4 | planned |
| People (grouping, naming, merge, split, forget) | [15](../photo_library/15_p2a2_people.md) | planned; grouping method: gap G7 |
| Relations between people | only "frequent pairs" person memories and an "often with me" weight (30 sections 4.1, 4.3); "me" is listed as later (00 section 2.2) | gap G18 |
| Scenes and subjects | detector classes as a filter; themes as prompt pairs over frame vectors (30 section 4.1); text and example-photo search | no stored scene tags; pets as identities are a non-goal (15 section 1): gap G19 |
| "On this day", featured photos, trip, event, person, holiday, birthday, theme and prompt memories | 30 sections 4.1 to 4.6 | planned |
| Selection, key photos, cover, titles and captions | 30 sections 4.4, 4.5 | planned |
| Rendering (slideshow timeline, framing, music) | 30 section 5 | planned |
| Search over all of it | 10 section 3 (`LibraryFilter`, the rule grammar), rule additions in 15 and 30; frame-target search (N9) | photo-level conjunctions owned; entity-level multi-hop questions (for example trips with a given person at a given place) have no read shape or index: gap G20 |

## 3. The order

1. **Now, beside each other:** N0's last two lanes on `int/native`; storage measurement round 3
   (M1 to M10) on the gate A spike; the fix lane for sweep 1115f's reds on `int/b4-stateless`, then
   its rerun and push; the rerun of the invalid ingest rows.
2. **N0 go** (G1 to G9 on plans F and C) merges `int/native`; N1's three lanes start; N2 starts
   when N1a is merged; N3 follows N2's green sweep. This is the critical path for speed.
3. **Gate B round 2** runs in a free hold as soon as possible: it is S3's and S5's entry.
4. **Storage waves A to G** on `int/storage`, each wave only when its entry measurements meet
   their bars: A (S1a-i, S1b-i, S1b-ii) after M1 to M8; B (S1a-ii, S2a, F); C (S2b, S3) after the
   owner's answer on the 0.6.0 upgrade code and gate B; D (S4, S5) after M9; E (S6, S7) after M10;
   F (S8a to S8c); G (S9: deletions, re-ingest, acceptance). The integration branch is merged into
   `int/storage` between waves.
5. **After N3:** the card budget (both cards, build settings), then the API ingest-path
   measurement (section 4). It also needs S1a-ii, so the port measures the final ingest write.
6. **After wave B:** write plan 16 (the labelling and training loop on the new store: SAM-like
   proposals, the few-labels head, train and promote); its lanes follow the slices they read.
7. **After waves B and C:** re-base the gallery plans on the record (G3); P2a-0 starts after that,
   then P2a-1, P2a-2, P2b, P2c, each after the previous exit bar and the owner's walk.
8. **After N3 and wave G:** core routes on the engine and the record (G5); then the rest of the API
   port by measurement; the release cut as the owner decides (section 9).
9. **After P2c:** video (P3, P4) and the desktop review.

```
int/native:  N0 ── go ── N1a ─┬─ N2 ── N3 ─┬─ card budget ── API ingest measurement ── port?
                    N1b, N1c ─┘            │                     ▲
int/storage: round 3 ─ A ─ B ─┬─ C ─ D ─ E ─ F ─ G ──────────────┤ (S1a-ii in B)
             gate B round 2 ──┘(S3, S5)    │    │                │
                                plan 16 ◄──┘ (after B)           core routes (after N3 and G)
gallery:     re-base plans (after B, C) ── P2a-0 ── P2a-1 ── P2a-2 ── P2b ── P2c ── P3, P4
```

**What must not start early** (the no write-then-rewrite rule: never write code a decided later
step rewrites):

| Do not start | Because | Until |
|---|---|---|
| Speed work on the Python frame engine | it is frozen as the reference and deleted at N3 | never (correctness fixes only) |
| New Python logic in the API's ingest path, beyond slice S1a-ii and fixes | the path is measured against the native engine and may be ported | the section 4 decision |
| Frontend work on wire fields that 15c section 2 changes (totals and caps, `join_truncated`, `rows_truncated`, image counts, sort ties) | each slice changes them with its screens | the slice that owns the screen |
| Gallery lanes against the search index (the `faces` index role of lane L-A0, the projection drain, the capture and `meta` fields) | 15 section 14, answer 3: they become claims and folded state | the re-base (G3) |
| Plan 13's search-index items (L1, L2, L4, L5, L6, V1, section 7) and 13a's L5 | the record replaces them (15 sections 5, 6, 9) | never: superseded |
| v0.8.0 features designed on Redis or search-index leases (10 F3, F5, F7, F9) | the control plane is PostgreSQL and Valkey | a re-base of plan 10 (G11) |
| New code on the core API's global search indexes | they move to project tables in the core-routes track | G5's plan |

## 4. The API port to Rust: order, measurement and how a port is judged

The order (owner, 2026-11-15): the frame engine first (plan 17); then the API's ingest path,
measured against the fast engine and ported if it is the limit; then the rest of the API, route
family by route family, by measurement. Training, export and the VLM stay Python: they are the GPU
frameworks' language. Storage schema and SQL carry over to a Rust API unchanged, so the storage work
proceeds now in Python.

| Step | Measured with | The bar for porting |
|---|---|---|
| Ingest path | after N3 and S1a-ii: the four-way ingest rows through the API (13 section 10), the per-stage request timeline, host CPU per photo per process (API workers, engine, PostgreSQL) and card busy share, on COCO and the 20 MP set, plan C | the API is the limit when its photos per second sit below min(engine capacity, card ceiling) of the same hold (17 section 0.2) by more than the spread while the card is not busy and the timeline puts the wait in API stages. A Rust ingest path then replaces it behind the same wire contract, passing the contract and database-tier tests unchanged, and is kept only if it wins by more than the spread |
| Other route families | the page bars of 15 section 6 (page shapes p50 50 ms, p99 200 ms at 8 clients) and the search latency, with host CPU per request | a family moves only when its measured profile shows the API process binding a number a user feels; contract tests pass unchanged |
| Training, export, the VLM | not a candidate | |

What keeps the port cheap, already the rule: the generated wire contract is the only boundary the
frontend knows; services carry no web-framework types; domain rules are pure and tested (their tests
are the port's specification); every model starts from ONNX; no store-specific detail reaches the
wire; SQL lives in the repositories. Every track records host CPU per photo per process next to its
throughput, so a port is decided by a table, not an opinion.

## 5. Gaps: what the goal needs and no plan owns, or plans that disagree

| # | Gap | Where | Proposed owner | Recommendation |
|---|---|---|---|---|
| G1 | The decided cluster rule (partition by detector class, larger clusters) has no lane; 13b claims clusters are computed within a detector class, the code does not | 13 line 502; 13b line 103; `src/services/clustering/run.py:42-43` | plan 16 | build it with slice S6 or right after it; correct 13b's sentence |
| G2 | Clustering fit time and memory at 10,000 to 1 million vectors are unmeasured | 13 line 502 ("Not measured") | plan 13 (a measured track) | measure before S6 and before any scale claim |
| G3 | The gallery plans assume the search index (faces role, projection drain, capture and `meta` fields, write-through with optimistic checks); 15 says they become claims and state, but no pass, owner or date is set | photo_library 00 lines 179, 210-250, 452, 658; 05 lines 109, 133-134, 170; 15 section 14 answer 3 | the photo-library plans, revised | one re-base pass after waves B and C, including the performance budgets (00 section 10) and the gallery's read shapes, which 15b's 80 shapes do not cover |
| G4 | Gallery engine needs without an owner: HEIC and RAW (N4), quality signals (N6), per-face signals (N7), moved files (N10); the N names are still provisional | photo_library 00 section 2.1; 17 line 610 (N1a keeps today's formats) | a plan 17 revision for N4, N6, N7; storage slice F for N10 | decide after N1; nothing in the gallery codes against them before |
| G5 | Core routes (/detect, /faces, /analyze, /search) on the engine and the record, and their global indexes moved to project tables: no plan file since revision 1's "phase 3" | 13 items A1, E3, E4 (lines 379-386); 15c answer 7 (`int/storage`, line 880) | a new plan 18 | write it when N3 and wave G are in sight |
| G6 | The labelling and training loop on the new store (SAM-like proposals, few-labels head, train, promote) was promised as plan 16 and never written | revision 1 of this file; 13 T5, T6, T10 | a new plan 16 | write it after wave B |
| G7 | Face grouping has two owners | 13 line 390 (F3); photo_library 15 lane L-P1 | photo_library 15, with 13's F3 numbers as its input | one face-identity service, built once, in P2a-2 or earlier if core /faces needs it |
| G8 | Plan 13 still says the Rust engine runs "with no model server between prep and inference, built on the owner's existing Rust serving code"; 17 keeps the model server and writes its own crates | 13 line 70; 17 sections 3 and 4.1 | plan 13 | mark 13 section 1a item 6 and track T1 superseded by 17 |
| G9 | The card-budget levers (P2, P5, P6, precision) were planned against the Python engine; 17 points at them, nobody re-plans them | 13 lines 374-378, section 8; 17 section 0a | plan 13, revision 3 | re-plan after N3 with 17's bars |
| G10 | The general GPU parts move into the owner's Rust server when the owner publishes it (decision of 2026-11-15, an N3 exit item in the coordinator's notes); 17 says only "as the copyright holder's act" and N3's exit does not list it. 17 also says the API "stays Python" | 17 lines 10, 378, 644-647 | plan 17 | add the move to N3's exit; scope line 10 to "not in this plan" |
| G11 | Plan 10 (v0.8.0) predates the record and the PostgreSQL control plane; its video (F7) overlaps gallery P3; its S3 backend (F10) overlaps 15's files layer | 10 lines 52-101 | plan 10, re-based | re-base before any v0.8.0 wave; one video plan |
| G12 | Video has two homes | 10 F7; photo_library 40 | photo_library 40 | plan video once, after P2c |
| G13 | Frame-target search and one row per photo have two homes | 10 F2; 09a; photo_library 00 N9 | slice S3 plus a short search plan | fold into S3's lane or a follow-up |
| G14 | No deploy plan for other machines and the cloud: managed PostgreSQL (with the vector extension, row-level security and the separate vector role of 15 section 14), object storage, GPU hosts; authentication (10 F5) is planned only for v0.8.0 and the API has none | 10 F5, F10; 15 section 8 | a new deploy plan | write it before the release that offers cloud; see question 4 |
| G15 | Plan 15 section 10 says the vector step "starts with the in-record index so nothing waits for gate B"; 15c makes gate B the entry of S3 and S5 | 15 line 234; 15c lines 58, 60 (`int/storage`) | plan 15 and 15c | run gate B now so the conflict does not bind; then align the text |
| G16 | Stale status text: the master plan's "where things stand" (2026-11-12), the vision's "next release" section, the desktop direction ("Python stays until then"; SQLite for the desktop, so the server's SQL does not carry over there unchanged); the project CLAUDE.md still calls the job lease an OpenSearch lease (`src/clients/job_lease.py` no longer exists) | 00 lines 64-87; VISION_AND_GOALS lines 128-170; photo_library 50 line 42; 15 line 160; CLAUDE.md "Singleton jobs" | the owners of those files | refresh 00 section 1, the vision and CLAUDE.md with this map; the desktop store is decided in its own plan after P2c |
| G17 | Records: the ingest reliability rerun passed but its row is not in 13 section 10; the earlier result rows predate the paper's evidence clause | 13 section 10; 17 section 5 | plan 13 | add the row; the paper's table is N3's, measured under the clause |
| G18 | Relations between people (household, often together, "me") are not modelled beyond memory scoring | photo_library 30 sections 4.1, 4.3; 00 section 2.2 | photo_library 15 or 30 | decide with the owner whether relations are in scope for P2a-2 or P2c; if so, an entity edge kind, not a graph store |
| G19 | Scenes and subjects have no stored tags (themes are computed per memory only) | photo_library 30 section 4.1; 15 section 1 | photo_library 10 (search) | decide whether scene tags are stored results (an analyser, rebuildable) or stay query-time themes |
| G20 | Entity-level multi-hop questions have no read shape, index or bar; the gallery's reads are not among 15b's 80 shapes | 15 section 4.5; photo_library 00 section 10 | the re-based gallery plans (G3) with a 15b addendum | list the gallery's read shapes, including two- and three-hop ones, and measure them like gate A |

## 6. Decisions that cross tracks

| Decision | Date, source | Consequence |
|---|---|---|
| Data may be wiped and re-ingested instead of migrated | owner, 2026-11-14; 15 header | the storage layout is designed clean; acceptance is a re-ingest |
| PostgreSQL is the record and the only store for curation screens; a search copy only as an additive fallback for a shape that misses its bar | owner 2026-11-14, coordinator 2026-11-15; 15 R1, R7; 15b section 1.4 | the search engine leaves the curation path in wave G; vectors live in the record, the index is derived |
| A person's decision is an appended claim, it pins what the person saw, claims rank by actor before time; machine results are rebuildable and carry their analyser version | 2026-11-15; 15 R2 to R5 | the lock rule is the shape of the data; a model change re-runs only rows below the new version |
| The frame engine is rewritten in Rust now; the Python engine is frozen as the reference and deleted at the switch; C and C++ where a library needs it | owner, 2026-11-15; 17 header | no two engines are kept; N3 is one change |
| The whole backend ends in Rust, in order: engine; the API's ingest path if it is the limit; the rest by measurement; training stays Python | owner, 2026-11-15; section 4 | section 4's measurement and bars |
| The general GPU parts are written here behind a strict boundary and move into the owner's Rust server (ortloom) when the owner publishes it | owner, 2026-11-15; 17 sections 2.0 and 4.7 | `gpu_frame` holds no plan, wire or product type (gap G10 for the exit item) |
| The owner intends to publish both and write a research paper on the measured speed | owner, 2026-11-15; 17 section 5 | every speed result is kept under 17's evidence clause; set P never appears in a published table |
| Storage schema and SQL carry over to a Rust API unchanged | owner and coordinator, 2026-11-15 | storage proceeds now |
| The crop encoder is its own model at a 224 input by default, 336 a deployment choice; no smaller encoder replaces it | 2026-11-15; 13b sections 6 to 9 | each size is its own vector space; switching size means a new project and a re-ingest |
| On one card the models bound the end-to-end rate; after the native engine the gains come from the card budget | 2026-11-15; 17 sections 0 and 0a | the engine is judged by engine capacity and by reaching the card's ceiling |
| This host's CPU is the baseline; nothing is tuned to it; two cards on large photos are host-bound here | owner, 2026-11-15; 17 header and section 0a | thread counts derive from cores; every result names the CPU |
| The model server stays for N0 to N3 | 2026-11-15; 17 section 4.1 | reopened only by a named measurement |
| Licences are recorded per model and library at selection; licence questions are deferred by the owner and are not gates | owner, 2026-11-14 and 2026-11-15; 17 Q3 | every result table has a licence column |
| A push of the integration branch follows a green full sweep; never `main`, a tag or a release without asking | owner, 2026-11-14 | the wave scripts refuse a moved head |

### 6.1 Measured facts (one table; every number from a tracked plan)

| Fact | Number | Source |
|---|---|---|
| Python engine limit with the crop cost taken away | about 70 photos/s on small photos, 47 on 20 MP, card half idle | 17 section 0; 13b section 6 |
| Input only (decode, letterbox, frame square), Rust pipeline against Python, one process | COCO 10,155 / 10,427 against 655 / 622 photos/s; 20 MP 226 / 232 (best 255.5 at 24 threads) against 178 / 171 | 17 section 0.1 |
| Detector + frame encoder end to end, Rust pipeline against Python | COCO 192 / 189 against 187 / 184 (models alone 192 to 197); 20 MP 127 / 124 against 123 / 119 | 17 section 0.1 |
| Host CPU per photo, end to end | COCO 7.3 to 7.7 ms against 14.4 to 14.7; 20 MP 92 to 103 against 106 to 108 | 17 section 0.1 |
| Rust pipeline correctness | model inputs bit-identical; 306 of 306 boxes identical; vectors cosine at least 0.99998 | 17 section 0.1 |
| GPU time per photo | decode about 3 ms at 20 MP; detector about 0.8 ms; frame encoder about 4.2 ms; crop encoder 2.1 to 2.3 ms per crop | 17 section 0a |
| Cores one card needs at 20 MP | 11 to 13; two cards 24 to 28, more than this host's 24 physical cores | 17 section 0a |
| Crop encoder at 224 against 336 | 2.1 to 2.3 against 4.5 to 4.8 ms per crop; COCO R@1 0.703 against 0.712; Cars 0.911 against 0.913; aircraft ten-label accuracy 0.747 against 0.775 | 13b sections 7 and 9 |
| Smaller crop encoders | 3 to 15 NMI points lost on birds or aircraft | 13b section 8 |
| Ingest through the API, merged code against the old code | set P 41.5 against 35.5 photos/s; COCO 25.2 against 25.7 (card 99% busy) | 13 section 10, rows "D" |
| A three-field change | 580 documents/s with the vector in the document, 11,604 without; 13,639 to 22,811 PostgreSQL rows/s | 15 section 1 |
| pgvector on 957,000 real vectors | p50 2.8 ms at 1 client; recall 0.997 at a 0.26% filter; all vectors read back at 69,630 rows/s | 15 section 1 |
| Gate A at 10 million objects (8 clients, p50 / p99) | deep list page 1.3 / 2 ms in PostgreSQL against 32.3 / 52 in the search engine; raw cluster cards 3,977 / 5,799 ms, hence summaries | 15b section 1.1 (`int/storage`) |
| Cluster rule on ground truth | 0.971 labelling decisions per item (shipped) against 0.707 (by detector class) | 13 section 10, row T7 |

## 7. Deploy

- One compose stack on a workstation is the reference. No service assumes another is on the same
  host: addresses come from configuration, health follows a probe, a restarted dependency is survived.
- Originals, derivatives and crops go through one storage layer (15 section 8, slice F), so a local
  directory and an object store are a configuration choice; no stored value is a host path.
- The database is reached through its client factories only; a managed PostgreSQL must provide the
  vector extension, row-level security and the separate vector role (15 section 14, answer 2).
- GPU sizes come from profiles built from measured memory (17 N2, item E4 of 13); the 12 GB card
  starts inside its memory. A second card is either one model server across both cards or one engine
  and model server per card: 17 N2's two-card row decides.
- Cluster orchestration is not planned. The API has no authentication and gets none in this
  programme (owner, 2026-11-15: added later, for local or cloud use). A deployment therefore keeps
  the API on loopback, a LAN or a private network, with the owner's consent; that is a deployment
  rule, not a feature to build (G14, question 4).

## 8. How the work runs

- The coordinator writes and decides the plans, reads the risky diffs, judges every measurement and
  orders the work. Every plan is a file in the repository before a lane starts; a plan that exists only
  in conversation does not exist.
- Lanes: one agent per lane in its own worktree and branch, one commit per concern, hooks on, never
  `--no-verify`, never push or merge. Light checks only (unit and contract tests seen failing first,
  hooks on touched files, the Rust gates). Database, compose, live, browser and GPU runs are listed as
  owed to the wave. At most three coding lanes per integration branch.
- One full sweep per batch of merged lanes, on the merged tree; what each merge owes the sweep is
  written down as it merges; reds go back to one fix lane. Builds happen once per sweep.
- GPU: every GPU run is the coordinator's hold on a named card after checking peer sessions' use of
  it (GPU 2 for N0; GPUs 0 and 2 for two-card rows). GPU 1 belongs to another project. Any host GPU
  process sets **both** `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES`, then prints and
  checks the device before work (incident of 2026-11-15: CUDA's default order put index 2 on GPU 1).
- The model and effort are named for every agent: the smallest for mechanical work, the middle one
  for lanes, high effort for lock-rule, concurrency, storage and GPU-memory work, the largest for plans
  and hard reviews.
- Evidence for the paper: every speed claim follows 17 section 5 (public set, hardware with the CPU,
  versions and digests, the exact command, the remaining bound, the Python engine in the same hold);
  raw files stay under the gitignored artifacts directory and are never bulk-deleted.
- A result that misses its bar is reported as a miss with its number; a bar moves only in writing,
  with the reason.
- In the owner's Rust server: a worktree and branch per lane, hooks on, no push without the owner's yes.

## 9. Questions for the owner (all four answered 2026-11-15; none open)

| # | Question | The owner's answer |
|---|---|---|
| 1 | Delete the 0.6.0 in-place upgrade code (15c, coordinator answer 6)? | Yes. 0.7.0 is a new build for speed; a 0.6.0 installation moves its work by exporting and importing labelled datasets. Storage wave C may start on this |
| 2 | A trained model whose training set held a photo erased later | Keep the model and record that a photo of its training set is missing. Expected to be rare, since exports are kept |
| 3 | What makes the 0.7.0 tag | No tag until the whole programme of this roadmap is done: the backend rewrite (engine, storage, the API where measured), the frontend updated for it with the gallery, the features on this map, and the milestone's issues closed as far as possible. Pushes of the integration branch after green sweeps continue; a tag does not |
| 4 | Authentication before a cloud deploy | Not now. The app runs without a login, locally or in the cloud, and authentication is added later. Speed, processing and the features come first. Until it exists the API stays unpublished beyond loopback, a LAN or a private network of the owner's choosing (section 7) |

Answered since the earlier lists, kept as a record: face tiles on large photos are built as F1
measured them, and faces are on by default in library projects only (owner, 2026-11-13); the
encoder whose repository runs its own code is moot (13b section 8 closed the smaller-encoder search);
the 336 crop build is a deployment choice (13b section 9); licences of `gpu_frame`, outside
contributions and the HEIC codec are deferred by the owner and are not gates (17 Q3); the gallery's
eight questions D1 and Q2 to Q8 were answered on 2026-11-13 ([90](../photo_library/90_owner_questions.md));
agents may branch in the owner's Rust server (2026-11-15). Seven research questions of the master
plan (section 9.3) stay open and are asked after the tag; none blocks a track here.
