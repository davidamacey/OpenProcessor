# 15: storage, one record and stores that can be rebuilt from it

| Field | Value |
|---|---|
| Status | plan, revision 0 (2026-11-15): the rules, the record model and the two measured gates. The field-by-field schema, the query map and the lane specs are companion files written from this one (section 12) |
| Code base read | `int/b4-stateless` at `4a6c8d788`, through a written inventory of every stored field, writer, query shape, table, human-decision path and file path |
| Refines | [14_roadmap.md](14_roadmap.md) phase 2; [13_backend_speed_plan.md](13_backend_speed_plan.md) decisions on storage and its vector-store track; [02a_postgres_schema.md](02a_postgres_schema.md) (the tables that exist) |
| Not in scope | what the models compute (phase 1b, phase 3); the gallery screens (phase 5). Their data shapes ARE in scope, so no later phase redesigns a table |
| Data | may be wiped and re-ingested (owner, 2026-11-14). No migration of search-index documents is written |

## 0. The user's job, and what storage has to do for it

A person adds photos. A detector finds the objects. The person labels their own classes cluster by
cluster and trains a model; later the same photos are a gallery. Storage has three duties:

1. Never lose or silently change something a person decided.
2. Let every machine result be thrown away and computed again, by a better model or after a fault,
   without touching what a person decided.
3. Answer the lists, counts and searches of the screens fast, with a write visible to the next read.

Today one search-index document per object carries all three kinds of data and its vector.

## 1. What was measured (2026-11-14, one run each, shared host)

| Fact | Number | Consequence |
|---|---|---|
| Changing 3 fields on a document that holds its vector | 580 documents per second | every label, cluster placement, duplicate group and score rewrite re-indexes the vector |
| The same change on a document without a vector | 11,604 per second | vectors and changing fields must not share a row or a document |
| The same change as a PostgreSQL row | 13,639 to 22,811 per second | the record can take the write load |
| pgvector (float16, HNSW) on 957,000 real vectors | unfiltered p50 2.8 ms at 1 client, 7.1 ms at 8; a filter keeping 0.26% of rows: recall 0.997 at 35 ms | PostgreSQL is a real candidate for the vector index at this size; nothing is known yet at 10 million |
| Reading every vector back from PostgreSQL | 69,630 rows per second | a clustering job reads 1 million vectors in 14 s from the record |
| Float16 against float32 vectors | search recall 0.992 and 0.999 of the float32 answer before any index; a clustering refit agrees 0.97 (two float32 seeds agree 0.72) | the kept copy of a vector is float16 |
| One item holds at most 74 boxes | a limit of nested documents | boxes are rows, not nested documents |

Counted in the code: 7 index roles, about 215 fields on the object document, 40 writers, 80 query
shapes (35 with aggregations, 12 nested, 5 vector searches), 80 PostgreSQL tables, 24 kinds of human
decision, 16 operations that must change more than one store. Four ways of keeping history exist
(two capped lists inside the document, a per-operation table of prior states, undo records).

## 2. Rules

| # | Rule |
|---|---|
| R1 | PostgreSQL is the record. A fact is written there first, in one transaction with every other fact of the same act. |
| R2 | Three kinds of row, never mixed in one table: what was added (photos and where they came from), what a person decided (claims), what a machine computed (results). |
| R3 | A human decision is one appended row. It is never updated and never deleted; an undo is another row. The current state of an object is computed from the machine results and the claims by one pure function. |
| R4 | A machine result names the analyser and the version that made it. A table of results can be emptied and filled again; rows below a new version are the work list of a re-run. |
| R5 | A machine writer cannot write a claim, and it never needs to read one to stay clear of it. The lock rule becomes the shape of the data instead of a check in every writer. |
| R6 | A vector lives in a table of its own, one row per subject and vector space, float16, with the model version beside it. Nothing else in that row changes. |
| R7 | Every other store (a vector index, a search copy, counters, cluster summaries, caches) is derived. Each has one owning module, one function that rebuilds it from the record, and one test that drops it, rebuilds it and compares answers. |
| R8 | Identity on the wire does not change (content-derived ids, class identity by name). Inside the record, rows join on compact integer keys. |
| R9 | A file is reached through the storage layer by key. No stored value is a host path. |
| R10 | Every project table carries the project and is closed by row-level security, as in 02a. Fail closed. |

## 3. Stores and owners

| Store | Holds | Rebuilt from | Decided by |
|---|---|---|---|
| PostgreSQL, record tables | photos, sources, claims, class registry, configuration, jobs, training and models | nothing: this is the backup target | fixed |
| PostgreSQL, result tables | detections, boxes, text, capture data, scores, cluster and duplicate runs, vectors | photos + models + configuration | fixed |
| PostgreSQL, current-state table | one row per object: the effective class, flags, geometry, placement | results + claims (section 4.4) | fixed |
| Vector index | approximate search over the vector tables | the vector tables | gate B (section 7) |
| List and facet reads | the 60 or so list, count and facet shapes | the current-state table | gate A (section 6): SQL on the record, or a search copy without vectors |
| Files | originals, uploads, exports, training artifacts; crops and thumbnails as a cache | originals: nothing; the rest: originals + record | section 8 |
| Valkey | queue transport and wake-ups only | nothing to rebuild | fixed; the dirty set of image counts goes |

## 4. The record model

Names here are working names; the companion schema file fixes them against the code.

### 4.1 Added: photos and sources

One row per stored photo per project: the integer key, the wire id (content-derived, unique), the
storage key of the original, size and hashes, pixel size, how it was decoded, when and by which
ingest run it arrived. Import and combine provenance are rows that point at the photo. The existing
content-claim table (`image_contents`) already makes one stored photo per content; it merges into
this table so there is one claim, taken by the insert itself.

### 4.2 Decided: the claim log

One table, append only. A row is: a rising id, the project, the time, the actor (a person, an
import, a bulk accept), the operation it belongs to, the kind, the subject, and a small payload.

| Part | Content |
|---|---|
| Subject | a photo, an object, a box, a pair (duplicate verdict), a face or a cluster. Always with the photo's wire id and, for an object or a box, its geometry at the time, so the claim can find its subject again after every machine result was rebuilt |
| Kinds | class set or cleared; validated; excluded or restored; object drawn; geometry changed; box created, moved, state, text; dismissed (review, VLM suggestion, probe); needs a new class; moved to a cluster; audit verdict; duplicate verdict; holdout frozen; person named or merged; and for the gallery: favourite, hidden, album membership, a date or place corrected |
| Class | by name, with the registry id of that moment as a hint only |
| Undo | a row that names the claim it withdraws. The current state is then the fold without that claim. No prior state is stored anywhere |
| Operations | the existing `label_operations` row groups the claims of one act; its member table and the two history lists inside documents are deleted |

A validated import label is a claim whose actor is the import. An unvalidated import label is a
machine opinion (4.3).

### 4.3 Computed: results

| Table family | One row per | Notes |
|---|---|---|
| Analysers | analyser kind, name, version, configuration hash | detector, region stage, text reader, VLM with prompt-pack revision, encoder build, capture parser, clustering method. The encoder rows are today's `embedding_model_versions` |
| Analysis ledger | photo and analyser kind: version done, state, time | "what has been computed for this photo"; the work list of any re-run; replaces the status fields scattered over the document |
| Capture data | photo | parser version; the timeline sort key and the map point |
| Objects (detections) | detected or drawn object | photo, box, detector class and confidence, analyser; a drawn object has no detector row and exists through its claim |
| Class opinions | object and source | detector, VLM, auto-label, probe, unvalidated import: class by name, confidence, analyser. The effective machine class is picked by one precedence function, not by whichever writer came last |
| Boxes | box inside an object | region-stage geometry, state proposed by the machine, detector, score; text readings as rows per reader |
| Faces, text lines | face, line | the same shape as objects; today's global face and text indexes are these tables under a project |
| Vectors | subject and space | R6. One table per vector space, created when the space is added (the `embedding_spaces` row already names it) |
| Runs with members | clustering partitions, duplicate runs, text consensus | exist today in PostgreSQL; their copies inside documents go (section 6) |

### 4.4 Current state

One narrow row per object holds what the lists filter and sort on: effective class and its source,
validated, excluded, holdout, effective geometry, rank in the photo, review flags, the times. It is
written in the same transaction as the claim or the result that changes it, by calling the one fold
function (`src/domain/`, pure, property-tested: any order of machine writes around a claim gives
the same row; a withdrawn claim leaves no trace; today's lock predicates in
`src/clients/occ_locks.py` hold on every generated history). A rebuild empties the table and folds
again. Because the write is transactional, a list read after a label sees it: no refresh wait, no
realtime get.

### 4.5 What the gallery adds (phase 5, tables designed now)

People, places, moments, trips and albums are entities; a photo's link to one is an edge row with a
kind, a weight and, when a machine made it, the analyser. Plain tables with indexes both ways; a
compressed bitmap of photo ids per entity for fast set arithmetic if the measurement asks for it.
No graph database: the questions are two or three hops over a few entity kinds, which joins answer.
A person's name and a merge are claims (4.2); the clustering of faces is a result (4.3).

## 5. Write paths

| Path | Today | With the record |
|---|---|---|
| Ingest of a photo | content claim, then the photo document, then the object documents, then events; a repair lease and an item list on the photo document cover a death between the writes | one transaction per commit group: photo, objects, opinions, vectors, capture, ledger, events. A photo is stored whole or not at all by construction. The repair code, the item list and the incomplete mark are deleted |
| A person labels, edits, undoes | read, merge, conditional write per document; history lists; prior-state rows | insert claims, fold, update current rows, one transaction; the bulk forms are the same statement over many subjects |
| Region stage, VLM, auto-label, probe, scores | conditional bulk writes that skip a document a person touched | upsert result rows; fold. No conflict with a person is possible (R5) |
| Clustering, duplicates | run tables in PostgreSQL, then a copy into every document, guarded by a fence and an outbox | run tables only; activation flips one pointer |
| Class rename, merge | registry row, outbox, rewrite of every document of the class | registry row; names are joined, never copied. A merge appends nothing to objects |
| Image counts | a dirty set in Valkey, a periodic recount into the photo document | a query, or a summary table kept by the writer if gate A asks for one |

## 6. Gate A: where lists, counts and facets are read

The question: can PostgreSQL also answer the interactive reads at 1 and 10 million objects per
project, or does a search copy (without vectors, fed from the record) stay?

Measured on the same generated rows in both engines, shape for shape: the object list with its
total, the review queue, cluster cards over a whole project, one cluster's facets, dataset
statistics, class counts, the photo page with object facts, box rows, locate, a text-substring
filter, the map grid, and reads while 1,000 objects per second are relabelled. Bars at 8 clients:
page shapes p50 at most 50 ms and p99 at most 200 ms; dashboard shapes p50 at most 200 ms and p99 at
most 800 ms. For the dashboard shapes PostgreSQL is also timed with small summary tables kept by the
writer.

| Outcome | Design |
|---|---|
| PostgreSQL meets the bars (with summary tables where needed) | one store. Every read is SQL behind the existing service functions; the search engine leaves the stack; a later single-file build of the app has one query language to port |
| PostgreSQL misses on a few dashboard shapes only | those shapes read summary tables; the rest is SQL |
| PostgreSQL misses broadly | a search copy without vectors stays as a derived store (R7), fed in commit order from a change table; today's query bodies keep working against it; writes still go to the record first |

Either way the vector searches move to the vector index (gate B), the nested box documents become
rows, and text search over box text and text lines is designed once for both (trigram index or the
search copy).

## 7. Gate B: the vector index

The record keeps every vector (R6), so the index is a derived store behind one small interface:
add or replace, delete, search with a filter, rebuild from the vector table. An index inside
PostgreSQL filters with the live rows themselves; an index outside it needs the filter fields copied
in, which is then part of that candidate's update cost.

First round, 957,000 vectors: recall at 10 with a 0.26% filter was 0.997 for pgvector and for
Milvus, 0.996 for VectorChord, 0.860 for a FAISS graph with an id selector. Latency is measured for
pgvector only so far.

Second round, decided before any default is chosen, at 10 million or more real vectors made by the
crop encoder that phase 1b selects: pgvector (graph on float16; graph on 1-bit codes with a float16
rescore), VectorChord, Milvus (graph, disk and GPU indexes), Qdrant, OpenSearch. Per candidate: build
time and peak memory, size on disk and resident, recall at 10 against exact search (unfiltered, a 3%
filter, a 0.3% filter), p50 and p99 at 1, 8 and 32 clients, inserts per second while searching,
the cost of changing a filter field on 100,000 rows, start to first answer after a restart, licence.
A candidate that needs a second service must beat the in-record candidate by more than the
run-to-run spread on a bar that a user feels.

Small sets do not use the index: when a filter leaves fewer rows than a threshold (measured), the
search is exact over those rows.

## 8. Files

- One storage layer (the existing `RootedStore` contract) for every read and write of an original,
  an upload, an export, a training artifact, a snapshot. A local directory and an object store are
  two implementations. The inventory lists the callers that open paths directly today; each moves.
- The photo row stores a key, never a path. Source roots that are read in place stay possible: the
  key names the root and the relative path.
- Crops and thumbnails are a cache keyed by photo id, box and size, written by the engine's
  derivative branch, with a size cap and no assumption of shared memory.
- Local-only calls (read-ahead hints, symlinks in exports) live inside the local implementation.

## 9. What this deletes

The repair lease and item list of plan 13a's L0; the conditional read-merge-write helpers for
project data and the lock checks inside writers; both history lists and the prior-state table; the
outbox and the fenced projections for classes, clusters and duplicates; the Valkey dirty set; the
nested box documents and their 74-box limit; mapped fields with no writer; the five global indexes
(their data becomes project tables in phase 3 and 5, designed here). Deleted in the lane that
replaces each, never left beside the new path.

## 10. Order of work

On a branch of its own off the integration branch, merged back when the acceptance of section 11
passes. At most three lanes at a time.

| Step | Work | Depends on |
|---|---|---|
| 1 | Schema migration; repositories; the fold function with its property tests; the analysers and ledger | this plan and the schema file |
| 2 | Ingest writes the record in one transaction per commit group (after 13a's per-photo flow) | 1 |
| 3 | Human write paths as claims: label, bulk label, undo, exclude, dismiss, geometry, drawn object, boxes, holdout, import | 1 |
| 4 | Machine writers as results: region stage, VLM, auto-label, probe, scores, projection coordinates | 1 |
| 5 | Reads, by family, per gate A | 1, gate A |
| 6 | Vector index behind its interface; the search routes; clustering and duplicate jobs read vectors from the record | 1, gate B for the default only |
| 7 | Files through the storage layer | independent |
| 8 | Deletions of section 9 not already done; re-ingest; acceptance | 2 to 7 |

Steps 3, 4 and 7 run beside each other. Step 6 starts with the in-record index so nothing waits
for gate B; the interface makes the default a configuration choice.

## 11. Acceptance

- Re-ingest of the public sets and set P on the new store; the four-way ingest table of plan 13 not
  below the numbers measured before the change; host CPU per photo per process recorded.
- For each derived store: drop, rebuild, same answers (R7).
- Empty every result table, re-run the analysers, replay nothing by hand: every claim is attached to
  its subject and the current state equals the state before.
- The scripted labelling walk (label by cluster, undo, bulk accept, exclude, freeze the holdout,
  export, train) passes on the public set with the same trained-model AP as before the change.
- The wire contract is unchanged except where a field had no writer.
- Two API processes and two workers writing the same objects under load: no lost claim, no
  current-state row that differs from a fresh fold.
- Gate A and gate B bars met by the chosen designs at 10 million objects.

## 12. Companion files, written from this plan before any lane starts

| File | Content |
|---|---|
| 15a, schema | every table with columns, keys, indexes and row-level security; every one of today's fields mapped to a column, a derived value or "deleted", with the reason; the claim kinds and payloads; the fold function's truth table against today's lock predicates |
| 15b, reads | each of the 80 query shapes mapped to its new query (both gate A outcomes), with the index it needs |
| 15c, lanes | lane specs for section 10 with keep-lists, tests seen failing first, and what each lane owes the sweep |

An independent review of 15 and 15a follows, then a fix pass, then lanes.

## 13. Open

| Question | Settled by |
|---|---|
| Gate A outcome | the read-shape measurement at 1 and 10 million rows (running) |
| Gate B default | the 10 million vector round, after the crop encoder is chosen |
| Whether cluster placement is a column of the current-state row or a side table swapped at activation | gate A's cluster-card numbers |
| Exact counts on very large filtered lists, or a capped count ("10,000+") | gate A's count numbers; a wire change if capped |
