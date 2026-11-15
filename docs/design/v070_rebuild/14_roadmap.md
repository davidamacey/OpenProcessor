# 14: roadmap from the ingest path to the full application

| Field | Value |
|---|---|
| Status | roadmap, revision 1 (2026-11-14); the owner's direction of that day, ordered by the coordinator |
| Reads with | [00_master_plan.md](00_master_plan.md) (the waves to the 0.7.0 tag), [13](13_backend_speed_plan.md) and [13a](13a_ingest_on_engine_plan.md) (speed), [the photo-library plan](../photo_library/00_photo_library_plan.md) (gallery), [the desktop direction](../photo_library/50_desktop_direction.md) |
| Rule | every phase gets its own plan file, an independent review, lanes, one test sweep, measured rows, a ledger entry. This file only fixes the order, the gates and the decisions that cross phases |

## 1. What is being built

One application. A person adds photos and everything a machine can say about a photo is computed
once on an NVIDIA GPU from one decode and stored once. On top of that store sit two things a
person uses: a gallery (browse, search, people, duplicates, moments, memories) and projects
(cluster, label, propose boxes for objects no detector knows, train a detector, use it). It runs
on one workstation, on other machines and in a cloud account with the same images. When the
Python design is settled, the hot paths move to Rust.

## 2. Phases and gates

| # | Phase | Plan file | Gate (measured, on public ground truth where one exists) |
|---|---|---|---|
| 1 | Ingest path: every photo through the frame engine, photos flow one by one to their write | [13a](13a_ingest_on_engine_plan.md) | not below the old code on any row of the four-way table; GPU busy at least 90% at 4 clients; full sweep green |
| 1b | The crop encoder: what is encoded, not how fast | 13 track T3 and its follow-up | crop retrieval and cluster quality not below today; the encoder no longer bounds ingest below 100 photos per second per card on paper, then measured |
| 2 | Storage: a clean layout, filled by a re-ingest | 15 (to be written when the bake-off and the document-split spike report) | every derived store dropped and rebuilt with equal answers; filtered search recall and update cost at the bars of 13 section 7, at 1 million and at 10 million or more vectors |
| 3 | One engine for every route: detect, faces, OCR, analyze, HEIC and RAW; old paths deleted; a 12 GB card profile | 13 items A1, E3, E4 (a plan file when phase 1 closes) | accuracy at or above today's on the public sets; one code path per branch; the small-card profile serves the full plan |
| 4 | Projects: cluster, label, propose, train, promote, reviewed end to end on the new store | 16 (to be written; starts from 13 rows T7 and T10 and the workflow audit [12](12_review_workflow_audit.md)) | a scripted acceptance walk on a public set and on set P, scored against ground truth: labelling decisions per item, detector AP of the trained model |
| 5 | Gallery, from the photo-library plans P2a to P2c | [photo library](../photo_library/00_photo_library_plan.md) | every screen opened in a browser at desktop and phone width; the plan's own per-phase checks |
| 6 | Deploy: one compose stack here, the same images elsewhere and in a cloud account | part of 15 and 16 | a clean install on a second machine; object storage and a managed database swapped in by configuration only |
| 7 | Rust: the engine first; more of the backend only where measured | 17 (to be written after phase 3) | section 4 |

Phases 4 and 5 are both wanted by the owner and share the store of phase 2. They run as two
tracks after phase 2, at most three lanes at a time in total. Video follows the gallery
([outline](../photo_library/40_p3_p4_video_outline.md)).

## 3. Decisions that cross phases

| Decision | Source | Consequence |
|---|---|---|
| Data may be wiped and re-ingested instead of migrated | owner, 2026-11-14 | phase 2 designs the layout clean; no backfill machinery beyond what a re-ingest needs; a simple additive migration stays fine where it is the short path |
| The store is chosen by measurement; PostgreSQL with extensions is a first-class candidate for everything | owner, 2026-11-14 | the bake-off answers "can it be the single store, with which limits"; the system of record and the search index are two decisions: one durable copy of every vector, the index derived from it |
| Machine results are rebuildable; a human decision is a row in an append-only claim log, replayed after any rebuild | 13 section 10 and the storage rules | labels leave the item documents before the storage change; the few-labels classifier and every cluster are derived artifacts |
| Every result carries the version of the analyser that made it | storage rules | a model change re-runs only the rows below the new version |
| The engine is judged by ground truth, not by bit-parity with the path it replaces | 13a, S1 | a new engine or encoder is accepted on detector AP, retrieval recall and cluster quality |
| Licences are recorded per model and library and decided at selection time | owner, 2026-11-14 | every result table has a licence column |
| A push of the integration branch follows a green full sweep | owner, 2026-11-14 | never a push of `main`, a tag or a release without asking |

## 4. Rust

What is known (measured on 2026-11-14): the Python engine spends 86 to 89 ms of host CPU per
small photo at 32 photos in flight and 37 to 42 ms at one, in one process. The image encoder
bounds ingest today (about 27 photos per second per card on COCO), so that cost is hidden. Once
phase 1b lifts the encoder bound, the engine process is the next bound. That makes the engine the
first Rust target, with a number to beat.

| Part | Position | Decided by |
|---|---|---|
| Frame engine (decode, kernels, model calls, branch fan-out) | Rust, behind the same protocol version 1, so the API cannot tell the difference | per plan (detect only, faces, full): photos per second, host CPU per photo, p50 and p99, memory, image size, start time, and the ground-truth gates of phase 3. Adopted where it wins by more than the run-to-run spread |
| Model serving | the engine may call the runtime in process (ONNX Runtime with TensorRT) instead of a model server | the same table, plus the kill tests: a decode fault must not take the models down |
| API, jobs, storage access | Python stays unless a measured profile shows the API process binding a user-visible number after phases 1 to 3 | request timelines of phase 1 and 2; a route moves only with its contract tests passing unchanged |
| Training, export, the VLM | Python (the training libraries are Python) | not a candidate |
| Desktop app | a later product on the same contract ([direction](../photo_library/50_desktop_direction.md)); a Rust engine and a Rust API make it one binary plus a bundled model runtime | after phase 5 |

What keeps the whole-backend option open at no cost now, and is already the rule: the generated
wire contract is the only boundary the frontend knows; services carry no web-framework types;
domain rules are pure and tested; every model starts from ONNX; no store-specific detail reaches
the wire. A second rule from this phase on: every phase records host CPU per photo per process
next to its throughput, so the case for or against a port is a table, not an opinion.

## 5. Deploy

- One compose stack on a workstation is the reference. Nothing in a service assumes the other
  services are on the same host: addresses come from configuration, health follows a probe, and
  a restart of any dependency is survived (the restart checks of 13 section 10).
- Originals, derivatives and crops are reached through one storage layer, so a local directory
  and an object store are a configuration choice (phase 2 writes the layer's contract).
- The database and the search index are reached through their client factories only, so a
  managed service replaces a container by configuration.
- GPU placement and sizes come from profiles; the engine fits every profile (13 item E4).
- Cluster-level orchestration is not planned; a second card or host is a second engine and model
  server pair behind the same API (13 decision D7).

## 6. How the work runs

- The coordinator writes and decides the plans, reads the risky diffs, judges every measurement
  and orders the work. Lanes implement from a written plan in their own worktree.
- At most three agents at once. A measurement runs as a bounded script that writes files; one
  short write-up follows. No agent waits on a queue.
- The model and the effort are named for every agent: the smallest for mechanical work, the
  middle one for lanes, high effort only for lock-rule, concurrency and migration work, the
  largest only for a plan.
- One full sweep per batch of merged lanes; what each merge owes the sweep is listed as it merges.
- A result that misses its bar is reported as a miss with its number. A bar changes only in
  writing, with the reason, as in 13a's S1.

## 7. Open, for the owner (none blocks phase 1)

| Question | Recommendation |
|---|---|
| HEIC: the CPU route measured 6.8 photos per second on 8 cores at 12 MP, the GPU video-decoder route 28 with a per-tile loop | build both: the CPU route as the fallback first, the GPU route with a batched decode in phase 3; the codec licence is decided at selection |
| Face tiles on large photos cost about 12 ms of GPU per 20 MP photo and find small faces (0.77 to 0.996 at about 110 px) | on for libraries with people, off by default in projects without a people class |
| A model whose repository must run its own code to load (one candidate encoder) | run it only inside a throwaway container with no network, if at all |
