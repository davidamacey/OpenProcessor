# 15a: storage schema, the tables of the record

| Field | Value |
|---|---|
| Status | plan, revision 1 (2026-11-15): the independent review and the coordinator's decisions on it folded in; appendix A traces each finding. Companion of [15_storage_plan.md](15_storage_plan.md) section 12; its decisions are fixed here, not reopened |
| Code base read | `lane/speed-plan` at `ca918ea2d` (integration code of `int/b4-stateless` plus plan 15); the storage inventory of 2026-11-13 (`artifacts_local/speed_1113/storage_inventory.md`, gitignored), re-checked in code where a design depends on it |
| Conventions | [02a_postgres_schema.md](02a_postgres_schema.md) section 1 and [02d_research_seam_schema.md](02d_research_seam_schema.md) section 0.1 apply to every table here, except where section 1 below says why not |
| Server | PostgreSQL 18 with pgvector 0.8.7 (`pgvector/pgvector:0.8.7-pg18-trixie`, [01_package_pins.md](01_package_pins.md); [pgvector README](https://github.com/pgvector/pgvector)) |
| Not here | the query map (15b), the lane specs (15c), what the models compute |
| Data | wiped and re-ingested (owner, 2026-11-14): no migration of search-index documents |

Plain words used throughout: a **photo** is a stored image (today the `images` document); an **item**
is a detected or drawn object (today the `items` document, wire id `crop_id`); a **box** is one region
inside an item (today one element of `region_boxes`); a **claim** is one human decision; a
**result** is one machine output; the **fold** turns results and live claims into the current state.

## 0. What this file fixes

It gives every table of the record with columns, keys and row-level security; maps every field
stored today to a column, a value derived at read, or a deletion; defines the claim kinds and their
payloads; defines the fold and proves today's lock rule from the shape of the data; defines erasure
and the wire fields that change. Indexes beyond keys come from gate A (section 2.6).

| 02a / 02d table | Fate | Section |
|---|---|---|
| `image_contents` | **deleted**: merged into `photos` (plan 15 section 4.1); its sorted imohash locks move into the ingest transaction; dropped by the ingest lane | 2.1, 7 |
| `label_operations` | **changed**: groups claims; keeps its stored status (`undoing`, `undone`, set by the undo job); gains `dataset_import_id`; single-label rows are pruned | 2.2 |
| `label_operation_members` | **deleted**: claims carry their operation; no prior state is stored | 2.2 |
| `near_dup_decisions` | **deleted**: a duplicate verdict is a claim | 3 |
| `search_projection_tasks` | **deleted**: no projection into documents remains (plan 15 section 9) | 9 |
| `embedding_spaces` | **changed**: the OpenSearch names (`index_role`, `vector_field`, `state_field`, `nested_path`, `knn_method`) give way to `vector_table` and `vector_index` | 6 |
| `cluster_memberships` | **changed**: `item_key text` becomes `subject_key bigint`; it is the placement every reader joins (plan 15 section 13) | 2.3 |
| `near_dup_members`, `near_dup_edges`, `holdout_near_dup_moves` | **changed**: text member ids become `bigint` subject keys | 2.3 |
| `dataset_imports` | **changed**: gains `dataset_name` (today `import_dataset_name` sits on every item) | 2.1 |
| `dataset_import_images` | **changed**: `image_id text` becomes `photo_key bigint`, `item_ids` goes (items of an import are its claims) | 2.1 |
| `project_indexes` | kept until gate A; deleted with the search engine if gate A ends in "one store" | 9 |
| every other 02a / 02d table (`projects`, `class_registries`, `project_classes`, `holdout_freezes`, `cluster_partitions`, `cluster_identities`, `partition_clusters`, `cluster_activations`, `near_dup_runs`, `near_dup_groups`, `near_dup_pending`, `embedding_model_versions`, `jobs`, `events`, ...) | **kept** as written | |

New tables: 35 created now (sections 2 and 8), of which the 10 gallery tables of section 2.5 are
filled first in phase 5; plus one vector table per embedding space, created at run time from one
template (section 6). Changed: 8. Deleted: 4, and `project_indexes` if gate A ends in one store.

## 1. Conventions used here

| Convention | Rule | Tie to 02a, and where it cannot hold |
|---|---|---|
| Keys inside | high-volume rows (photos, items, boxes, faces, detections, claims) have `bigint GENERATED ALWAYS AS IDENTITY` keys ([identity columns](https://www.postgresql.org/docs/18/ddl-identity-columns.html)); low-volume entity rows keep `uuid DEFAULT uuidv7()` | 02a 1.1 says entity keys are `uuidv7()`. At 10 million items a `uuid` costs 16 bytes in the row and in every index and child row, against 8; the locality argument of 02a (inserts stay at the right edge of the B-tree) holds for an identity column too. R8 of plan 15 asks for compact integers. This is the one 02a convention this file changes |
| Wire ids | content-derived 128-bit ids (`image_id` = `sha256(imohash)[:32]`, `src/services/near_duplicates/content_key.py:94`; `crop_id` = `sha256(image_id\|bbox)[:32]`, `src/services/detection/geometry.py:61`; drawn `crop_id`, `src/domain/item_geometry.py:75`) are stored in a `uuid` column, unique per project beside the integer key. PostgreSQL accepts 32 hex digits without hyphens as `uuid` input ([UUID type](https://www.postgresql.org/docs/18/datatype-uuid.html)); one helper `src/domain/wire_ids.py` renders the value back as 32 lowercase hex, so the wire is unchanged. 16 bytes instead of 33 for `text` | new; 02a has no wire-id column of this size |
| Box ids | `box_id text` `b<N>`, minted from `items.box_seq` (today `region_box_seq`, `src/domain/region_boxes.py:133`) | unchanged rule |
| Project column | every table that holds project data has `project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE` and the policy below; repositories always add `project_id = :p` (02a 1.4, RC-9) | 02a 1.2, 1.4. Child tables keep a direct `project_id` instead of the `EXISTS` policy of 02a section 5, because they are unbounded (the X-1 rule of 02d section 3) |
| Policy | `ENABLE` and `FORCE ROW LEVEL SECURITY`, policy `p_<t>_project` exactly as 02a 1.4 (the InitPlan form of `op_scope_*`), called **RLS(02a)**. In the DDL below `PROJECT` stands for `project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE`, and **every table with a `project_id` column has RLS(02a)**; only `analysers` and `claim_kinds` (deployment catalogs) have none | 02a 1.4; the test `tests/db/test_row_level_security.py` already iterates every `project_id` table |
| No `project_id IS NULL` | none of the new tables allows a NULL project; the core API's data gets a project (plan 15 section 14, answer 1) | 02a 1.4 allows NULL rows under `core` scope for a few tables; a nullable column in every key at 10 million rows is not worth it |
| Names | tables plural snake_case; constraint names from the `MetaData` naming convention (02a 1.1); every closed set is `text` + named `CHECK` with one Python source and a test, except the claim kinds: a lookup table with a foreign key (section 2.2), because adding a value to a `CHECK` on a 20 M-row table scans it under `ACCESS EXCLUSIVE` unless done `NOT VALID` then `VALIDATE` ([ALTER TABLE](https://www.postgresql.org/docs/18/sql-altertable.html)) | 02a 1.1, RR-17 |
| Times | `timestamptz NOT NULL DEFAULT now()`; capture times stay floating `timestamp` (the camera's local wall clock), as the `capture` mapping stores them today (`src/clients/curation_opensearch/capture_fields.py:35`) | 02a 1.1 |
| Actors | `*_by text` as 02a 1.1 (`api`, `worker:<host>`, ...); claims add `actor_kind` | 02a 1.1 |
| `jsonb` | only for bodies validated by a pydantic model on write and read: claim payloads, assist answers, analyser parameters | 02a 1.1, D-59 |
| No partitioning | no table here is partitioned. Partition by project would make `project_delete` a `DROP`, but creating a partition is DDL and `op_app` has none (02a 1.5); the planner handles "up to a few thousand partitions fairly well" only with pruning ([partitioning](https://www.postgresql.org/docs/18/ddl-partitioning.html)). The batched delete of 02a 1.2 stays | 02a 1.2, 1.5 |
| Bulk writes | `COPY FROM` is refused on a table with row-level security ("Currently, COPY FROM is not supported for tables with row-level security", [COPY](https://www.postgresql.org/docs/18/sql-copy.html)). Bulk writers use the pattern already in `src/services/clustering/partitions.py:30-44`: `COPY` into an `ON COMMIT DROP` temporary table, then one `INSERT ... SELECT` that the policy checks | the 02d note "bulk-written with COPY" means this pattern |
| Updated rows | a HOT update needs "no indexed column modified" and "sufficient free space on the page" ([HOT](https://www.postgresql.org/docs/18/storage-hot.html)); which columns are indexed and each table's `fillfactor` (10-100, default 100, [CREATE TABLE](https://www.postgresql.org/docs/18/sql-createtable.html)) come from gate A (section 2.6). Every fold write is `UPDATE ... WHERE (cols) IS DISTINCT FROM (new)`, so a no-op writes no tuple | new |
| Indexes | section 2 creates primary keys and unique keys, and every foreign key whose referenced row can be deleted gets an index on its referencing column, `ix_<table>_<column>`, partial `WHERE <column> IS NOT NULL` when nullable (cascades, `SET NULL` and erasure need it); the migration generates these from the metadata, the schema test asserts them, and the DDL below writes out only those with an extra key column. `analyser_id` references get none: `analysers` rows are immutable and never deleted. Every other index is in section 2.6 | new |

Size arithmetic: an 8 kB page holds 8,168 bytes of tuples; a tuple costs a 24-byte header (23
rounded), its columns with alignment, and a 4-byte line pointer. "10 M" below means a project with
10 million items, about 3 million photos (3.3 items per photo, the set P ratio of plan 13), 5 million
boxes and 20 million claims; each figure shows its arithmetic. Heaps are given at fillfactor 100;
index sizes are re-computed after gate A, when the index set is known.

## 2. Tables

### 2.1 Added: photos and where they came from

```sql
CREATE TABLE photos (                       -- one stored photo per content per project; replaces image_contents and the images document
  photo_key          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,
  image_id           uuid NOT NULL,         -- wire id, sha256(imohash)[:32]
  imohash            bytea NOT NULL CHECK (length(imohash) = 16),
  content_sha256     bytea NULL CHECK (length(content_sha256) = 32),   -- filled only to confirm an imohash match (RD-43)
  storage_key        text NOT NULL,         -- '<root>/<relative path>' through RootedStore (R9); never a host path
  size_bytes         bigint NOT NULL CHECK (size_bytes >= 0),
  width              integer NOT NULL CHECK (width > 0),
  height             integer NOT NULL CHECK (height > 0),
  frame              smallint NOT NULL CHECK (frame BETWEEN 1 AND 8),   -- EXIF orientation the decoder applied: the frame of every normalised box (3.4)
  source_state       text NOT NULL DEFAULT 'present' CHECK (source_state IN ('present','moved','missing')),   -- photo_library 00 N10
  decode_route       text NOT NULL,         -- 'gpu' or 'cpu_<reason>' (src/domain/decode_route.py)
  source             text NOT NULL,         -- the ingest source tag
  source_identifier  text NULL,             -- client identifier of a byte upload (BA-1)
  ingest_run_tag     text NULL,             -- client tag of one upload call (today ingest_run_id, BA-4)
  ingest_job_id      uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  ingest_request_id  text NULL,             -- X-Request-ID of the ingest call (today items.request_id)
  client_metadata    jsonb NULL,            -- core API caller metadata (today the global index 'metadata')
  created_at         timestamptz NOT NULL DEFAULT now(),   -- today indexed_at
  UNIQUE (project_id, image_id)
);
CREATE UNIQUE INDEX uq_photos_sha256 ON photos (project_id, content_sha256) WHERE content_sha256 IS NOT NULL;
-- The ingest duplicate check needs no imohash index: image_id is derived from the imohash, so the unique key
-- answers it; a confirmed collision's id derives from its SHA-256 (content_key.py:8-13), probed the same way.
-- Updated only to fill content_sha256 once and by the source scanner (storage_key repointed, source_state).
CREATE TABLE photo_provenance (             -- import and combine provenance; one row per photo and source
  provenance_key   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  photo_key        bigint NOT NULL REFERENCES photos(photo_key) ON DELETE CASCADE, PROJECT,
  kind             text NOT NULL CONSTRAINT ck_photo_provenance_kind CHECK (kind IN ('import','combine')),
  import_id        uuid NULL REFERENCES dataset_imports(id) ON DELETE CASCADE,
  combine_job_id   uuid NULL REFERENCES combine_runs(job_id) ON DELETE SET NULL,
  dataset_split    text NULL,               -- dataset_split (import) or origin_split (combine)
  source_stem      text NULL,  stratum text NULL,  hard_negative boolean NOT NULL DEFAULT false,
  label_state      text NULL CHECK (label_state IN ('labeled','negative','unlabeled')),
  negative_for     text[] NOT NULL DEFAULT '{}',   -- class NAMES (class identity is the name)
  origin_project_slug text NULL,  origin_image_id uuid NULL,
  created_at       timestamptz NOT NULL DEFAULT now(),
  CHECK ((kind = 'import') = (import_id IS NOT NULL))
);
CREATE UNIQUE INDEX uq_photo_provenance_source ON photo_provenance (photo_key, kind, import_id, origin_project_slug, origin_image_id) NULLS NOT DISTINCT;
  -- stable columns only: combine_job_id is SET NULL when the job goes, so it cannot be part of the key (CREATE INDEX, NULLS NOT DISTINCT)
```

`dataset_imports` gains `dataset_name text NULL`; `dataset_import_images.image_id text` becomes
`photo_key bigint NULL REFERENCES photos ON DELETE SET NULL` and `item_ids` is dropped.

Sizes at 10 M (3 M photos): `photos` tuple 24 + 8 + 16 + 16 + 17 + 33 (storage key, about 60
characters) ... about 224 bytes with text, so 3 M x 228 / 8,168 = 84 k pages, **0.7 GB**.
`photo_provenance` only for imported or combined photos: at most 3 M x 120 B = 0.36 GB.

### 2.2 Decided: claims and operations

The claim log is section 3. Its DDL:

```sql
CREATE TABLE claim_kinds (                   -- deployment catalog (no project_id, no RLS); rows = src/domain/claim_kinds.py (test);
  kind           text PRIMARY KEY,           -- a new kind is one INSERT in a migration, never a table scan
  subject_kind   text NOT NULL CHECK (subject_kind IN ('photo','media','item','box','pair','face','entity','claim')),
  layer          text NOT NULL,              -- fold layer of 4.3: geometry, class, flag, box, region, placement, photo, entity, face, withdraw
  pins_geometry  boolean NOT NULL,           -- R3 pin (4.3 layer 1)
  asserts_object boolean NOT NULL,           -- an unmatched claim creates an item (3.4 step 3); false: it stays dormant
  UNIQUE (kind, subject_kind)
);
CREATE TABLE claims (                        -- append only: no UPDATE, no DELETE (trigger claims_append_only)
  claim_id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,   -- the rising id; claim order within a rank
  at                  timestamptz NOT NULL DEFAULT now(),
  actor_kind          text NOT NULL CONSTRAINT ck_claims_actor_kind CHECK (actor_kind IN ('human','import','combine')),
  actor               text NOT NULL,         -- 'api', 'import:<id>', 'combine:<job>' (02a 1.1)
  actor_user_id       uuid NULL,             -- v0.8.0 authentication; no FK now (02a 1.1)
  op_id               uuid NULL,             -- the label_operations row; no FK: single-label rows are pruned (3.5) and a claim is never updated
  kind                text NOT NULL,
  subject_kind        text NOT NULL,
  photo_key           bigint NULL REFERENCES photos(photo_key) ON DELETE CASCADE,
  subject_crop_id     uuid NULL,             -- the item's wire id when the claim was made; a pair's first member at item level
  subject_box_id      text NULL,
  subject_face_id     uuid NULL, subject_pair_id uuid NULL,   -- faces.face_id; a pair's second member (image or crop wire id)
  subject_video_key   bigint NULL,           -- 'media' on a video (photo_library 00 library_videos; FK added by its lane)
  subject_entity_key  bigint NULL REFERENCES entities(entity_key) ON DELETE CASCADE,
  subject_bbox        real[] NULL CHECK (subject_bbox IS NULL OR cardinality(subject_bbox) = 4),          -- item or box geometry the person saw
  subject_machine_bbox real[] NULL CHECK (subject_machine_bbox IS NULL OR cardinality(subject_machine_bbox) = 4), -- the detector box at the time
  subject_frame       smallint NULL,         -- photos.frame when the claim was made (3.4)
  subject_key_hint    bigint NULL,           -- item_key / box_key / face_key at write time: a fast path, never authoritative
  withdraws_claim_id  bigint NULL REFERENCES claims(claim_id),
  payload             jsonb NOT NULL DEFAULT '{}',   -- pydantic model per kind (section 3.3)
  FOREIGN KEY (kind, subject_kind) REFERENCES claim_kinds (kind, subject_kind),
  CHECK ((kind = 'withdraw') = (withdraws_claim_id IS NOT NULL)),
  CHECK (subject_kind NOT IN ('item','box') OR (photo_key IS NOT NULL AND subject_crop_id IS NOT NULL
         AND subject_bbox IS NOT NULL AND subject_frame IS NOT NULL)),
  CHECK (subject_kind <> 'box' OR subject_box_id IS NOT NULL),
  CHECK (subject_kind <> 'media' OR (photo_key IS NULL) <> (subject_video_key IS NULL))
);
CREATE INDEX ix_claims_photo ON claims (photo_key, claim_id) WHERE photo_key IS NOT NULL;   -- FK; the fold of one photo's items
CREATE INDEX ix_claims_entity ON claims (subject_entity_key, claim_id) WHERE subject_entity_key IS NOT NULL;   -- FK
CREATE UNIQUE INDEX uq_claims_withdraws ON claims (withdraws_claim_id) WHERE withdraws_claim_id IS NOT NULL;  -- a claim is withdrawn once
CREATE UNIQUE INDEX uq_claims_op_subject ON claims (op_id, kind, photo_key, subject_crop_id, subject_box_id,
       subject_face_id, subject_pair_id, subject_video_key, subject_entity_key) NULLS NOT DISTINCT
       WHERE op_id IS NOT NULL AND kind <> 'withdraw';
  -- M4: one claim per operation, kind and whole subject; a resumed chunk inserts ON CONFLICT DO NOTHING ("conflict_target
  -- can be omitted", INSERT page) and writes nothing twice. Leading op_id, it is also the undo's read of an operation.
-- fillfactor 100 (never updated). Trigger claims_append_only: BEFORE UPDATE OR DELETE raises, except a DELETE
-- under op_project_delete_allows(OLD.project_id) (02a 1.2, RN-1) or op_erasure_allows(OLD.project_id) (section 8).
-- 02a label_operations, changed: it groups the claims of one act and keeps its stored status
ALTER TABLE label_operations ADD COLUMN dataset_import_id uuid NULL REFERENCES dataset_imports(id) ON DELETE SET NULL;
-- status keeps ('writing','written','undoing','undone','failed'): the undo job sets undoing, then undone and undone_at
-- (3.5). kind (src/domain/label_ops.py): label, batch_label, move, accept_machine, adopt_detector, unlabel, discard,
-- exclude, unexclude, dismiss, region_edit, geometry, draw, holdout_freeze, audit_start, dataset_import, combine,
-- assist_loop_propagate, duplicate_verdict, undo, and the phase-5 gallery kinds
DROP TABLE label_operation_members;
```

Sizes at 10 M: a claim tuple is 24 + 8 + 16 + 8 + 6 + 4 + 16 + 12 (kind) + 6 + 8 + 16 + 2 + 40
(one `real[4]`: 24-byte array header + 16; `subject_machine_bbox` and the other subject columns are
NULL on most rows) + 8 + about 125 (a `class_set` payload as `jsonb`: class name, hint, label
source, via, labeller, profile) = about 300 bytes; 20 M x 304 / 8,168 = 744 k pages, **6.1 GB**.

### 2.3 Computed: analysers, ledger and results

A row with `op_id` or `import_id` set (import suggestion, adopted or propagated class, an import's
box or detection) is **not rebuildable** (R2): re-runs and "empty every result table" keep it; only
the undo of its operation or import deletes it.

```sql
CREATE TABLE analysers (                     -- deployment catalog of everything that computes: no project_id, no RLS
  analyser_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  kind text NOT NULL CONSTRAINT ck_analysers_kind CHECK (kind IN ('detector','open_vocab','region_stage','text_reader','vlm',
       'classifier','probe','score','projection','capture_parser','encoder','face_detector','assist','raw_label_clustering',
       'import')),                           -- import: name = dataset name, version = dataset sha256 (an import's boxes and detections)
  name text NOT NULL,                        -- detector model, VLM endpoint 'name@revision', 'capture', probe checkpoint, ...
  version text NOT NULL,                     -- detector_version, VLM model, 'capture@1', method version
  config_sha256 text NOT NULL,               -- canonical JSON of params: prompt pack revision, region profile revision, ...
  config_ref jsonb NOT NULL DEFAULT '{}',    -- {config_id, revision} of a config document, or {model_version_id}
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (kind, name, version, config_sha256)
);  -- immutable (trigger analysers_immutable): a new prompt-pack revision or model build is a new row
CREATE TABLE analysis_ledger (               -- "what has been computed for this photo": THE work list of any re-run (m7)
  photo_key bigint NOT NULL REFERENCES photos(photo_key) ON DELETE CASCADE, PROJECT,
  kind text NOT NULL,                        -- analyser kind; 'encoder:<space_key>' per vector space
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),
  state text NOT NULL CONSTRAINT ck_analysis_ledger_state CHECK (state IN ('pending','done','failed','skipped')),
  reason text NULL, updated_at timestamptz NOT NULL DEFAULT now(),   -- reason: today's open_vocab_status and failures
  PRIMARY KEY (photo_key, kind)
);  -- the work list is "analyser_id <> the project's current analyser of the kind, or state <> 'done'": equality,
    -- never "below", because a configuration that returns to an earlier state reuses the earlier analyser row (m2)
CREATE TABLE photo_capture (                 -- capture data read from the file
  photo_key bigint PRIMARY KEY REFERENCES photos(photo_key) ON DELETE CASCADE, PROJECT,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),   -- today capture.parser 'capture@1'
  taken_local timestamp NULL, taken_utc timestamp NULL, offset_minutes smallint NULL,
  time_source text NULL, offset_source text NULL, time_precision text NULL,
  camera_make text NULL, camera_model text NULL, lens_model text NULL,
  focal_length_mm real NULL, focal_length_35mm real NULL, orientation smallint NULL,
  exif_status text NOT NULL, has_location boolean NOT NULL DEFAULT false,
  lat double precision NULL, lon double precision NULL, altitude_m real NULL,   -- positions only under the location policy (RD-51)
  location_accuracy_m real NULL, heading_deg real NULL, location_precision text NULL,
  burst_id text NULL,                        -- Apple BurstUUID and other makers' burst tags (photo_library 05 section 2.5)
  CHECK ((lat IS NULL) = (lon IS NULL))
);
CREATE TABLE items (                         -- identity of every item; stable across analyser re-runs (section 4.1)
  item_key bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,
  photo_key bigint NOT NULL REFERENCES photos(photo_key) ON DELETE CASCADE,
  crop_id uuid NOT NULL, box_seq integer NOT NULL DEFAULT 0,   -- box_seq: today region_box_seq, never decreases
  geometry_seq integer NOT NULL DEFAULT 0,   -- +1 by every result write that changes the item's machine box (4.3 layer 6)
  region_seq integer NOT NULL DEFAULT 0,     -- +1 by every region result applied; record data, so a rebuild keeps it (M11)
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (project_id, crop_id)
);                                           -- the counters are not indexed: their updates can be HOT
-- 0011 adds sort_area real NOT NULL, sort_rank smallint NOT NULL: copies of item_state.area and rank,
-- written by state.store_states only when they change (geometry, exclusion, attach), so the sort
-- indexes X29 and X30 live where a label writes nothing (2.6; 15b section 7)
CREATE TABLE detections (                    -- one detector box; replaced when the detector re-runs
  detection_key bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,
  photo_key bigint NOT NULL REFERENCES photos(photo_key) ON DELETE CASCADE,
  item_key bigint NOT NULL REFERENCES items(item_key) ON DELETE CASCADE,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),   -- detector, open-vocabulary set@revision, or import
  bbox real[] NOT NULL CHECK (cardinality(bbox) = 4),
  class_name text NULL, class_id integer NULL, confidence real NULL, proposal_name text NULL,   -- detector vocabulary
  blur_lap_var real NULL, blur_lap_ratio real NULL, blur_full_var real NULL,
  prompt text NULL, mask_polygon real[] NULL,                         -- open vocabulary
  import_id uuid NULL REFERENCES dataset_imports(id) ON DELETE CASCADE,   -- an import's proposed box: not rebuildable
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (item_key, analyser_id)
);
CREATE TABLE class_opinions (                -- machine class opinions: one per item and source (plan 15 section 4.3)
  item_key bigint NOT NULL REFERENCES items(item_key) ON DELETE CASCADE, PROJECT,
  source text NOT NULL CONSTRAINT ck_class_opinions_source CHECK (source IN
         ('classifier','vlm','assist_propagated','import_suggestion','adopted_detector','open_vocab_target')),
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),
  class_name text NULL, class_id_hint integer NULL,   -- NULL name: no registry class (vlm_unmatched)
  outcome text NOT NULL CONSTRAINT ck_class_opinions_outcome CHECK (outcome IN
          ('matched','unmatched','new_class_pending','reclassified','empty','low_conf','model')),
  confidence real NULL, confidence_cat text NULL,      -- classifier score; VLM category (vlm_confidence)
  raw_label text NULL, raw_label_conf real NULL, raw_class text NULL, proposed_class text NULL,
  empty_reason text NULL, via text NULL,               -- via 'combined': class and region in one VLM call
  op_id uuid NULL,                                     -- adoption and propagation: undo deletes the op's rows (no FK, as claims.op_id)
  import_id uuid NULL REFERENCES dataset_imports(id) ON DELETE CASCADE,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (item_key, source),
  CHECK ((source = 'import_suggestion') = (import_id IS NOT NULL)),
  CHECK (source NOT IN ('assist_propagated','adopted_detector') OR op_id IS NOT NULL)
);  -- a VLM answer without a class never replaces a row that has one (ON CONFLICT ... DO UPDATE WHERE
    -- EXCLUDED.class_name IS NOT NULL OR class_opinions.class_name IS NULL); the attempt goes to the ledger (M17)
CREATE TABLE raw_label_clusters (            -- clustering of VLM raw label strings (today vlm_label_cluster_* on every item)
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE, raw_label text NOT NULL,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),
  cluster_id integer NOT NULL, cluster_name text NOT NULL, distance real NULL,
  PRIMARY KEY (project_id, raw_label)
);
CREATE TABLE item_region_results (           -- the region stage's item-level outcome; replaced on re-run
  item_key bigint PRIMARY KEY REFERENCES items(item_key) ON DELETE CASCADE, PROJECT,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),   -- region profile name + revision, detector chain
  status text NOT NULL, reason text NULL,    -- machine region_status (RegionStatus values)
  verified boolean NULL, verified_at timestamptz NULL, verifier text NULL, verifier_version text NULL,
  auto_confirmed boolean NULL, visible boolean NULL, skip_verify boolean NULL, gate_skip text NULL,
  detector_chain text[] NOT NULL DEFAULT '{}', set_complete boolean NULL, detected_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE region_boxes (                  -- machine boxes of the region stage, and an import's proposed boxes
  box_key bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,
  item_key bigint NOT NULL REFERENCES items(item_key) ON DELETE CASCADE, box_id text NOT NULL,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),
  region_seq integer NOT NULL,               -- items.region_seq of the write that made it (4.3 layer 4)
  bbox real[] NOT NULL CHECK (cardinality(bbox) = 4),
  state text NOT NULL CHECK (state IN ('proposed','accepted','rejected','false_positive')),   -- BOX_STATES
  score real NULL, detector text NULL, detector_version text NULL, source text NULL,
  confidence text NULL, rejection_reason text NULL, detected_at timestamptz NOT NULL DEFAULT now(),
  import_id uuid NULL REFERENCES dataset_imports(id) ON DELETE CASCADE,
  UNIQUE (item_key, box_id)
);
CREATE TABLE box_texts (                     -- one reading per box and reader
  box_key bigint NOT NULL REFERENCES region_boxes(box_key) ON DELETE CASCADE, PROJECT,
  reader text NOT NULL CHECK (reader IN ('ocr','vlm')),
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),   -- text_reader, text_reader_version
  text text NULL, text_raw text NULL, text_frame text NULL, invalid text NULL, confidence real NULL,
  conf_char_min real NULL, jurisdiction text NULL, jurisdiction_conf real NULL,
  PRIMARY KEY (box_key, reader)
);
CREATE TABLE text_lines (                    -- OCR lines: whole photo (today visual_search_ocr) or item crop (item_text_lines)
  line_key bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,
  photo_key bigint NOT NULL REFERENCES photos(photo_key) ON DELETE CASCADE,
  item_key bigint NULL REFERENCES items(item_key) ON DELETE CASCADE,   -- NULL: a whole-photo line
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),
  seq smallint NOT NULL, text text NOT NULL, text_raw text NULL, box_norm real[] NOT NULL, polygon real[] NULL,
  det_score real NULL, rec_score real NULL, rel_height real NULL, language text NULL
);
CREATE TABLE probe_predictions (item_key bigint PRIMARY KEY REFERENCES items(item_key) ON DELETE CASCADE, PROJECT,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),   -- probe_model_version; params hold its class names
  class_name text NULL, class_id_hint integer NULL, entropy real NULL, confidence real NULL, margin real NULL,
  scored_at timestamptz NOT NULL DEFAULT now());   -- confidence feeds the wire probe_actionable (wire.py:253)
CREATE TABLE item_scores (item_key bigint NOT NULL REFERENCES items(item_key) ON DELETE CASCADE, PROJECT,
  kind text NOT NULL CHECK (kind IN ('uniqueness','mistakenness')),
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),   -- *_method, *_version
  score real NOT NULL, scored_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (item_key, kind));
CREATE TABLE item_projections (item_key bigint PRIMARY KEY REFERENCES items(item_key) ON DELETE CASCADE, PROJECT,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),   -- viz_projection_version
  x real NOT NULL, y real NOT NULL);
CREATE TABLE assist_answers (item_key bigint PRIMARY KEY REFERENCES items(item_key) ON DELETE CASCADE, PROJECT,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),   -- endpoint, model, prompt pack + revision
  run_id uuid NOT NULL, run_created_at timestamptz NOT NULL, class_name text NULL, class_confidence text NULL,
  attributes jsonb NOT NULL DEFAULT '[]', box_attributes jsonb NOT NULL DEFAULT '[]');   -- AssistAnswerBody
CREATE TABLE item_notes (                    -- re-detect and import notes on an item (proposal_merge.py:10-14, :32-40)
  note_key bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,
  item_key bigint NOT NULL REFERENCES items(item_key) ON DELETE CASCADE,
  kind text NOT NULL CHECK (kind IN ('proposal','class_mismatch','missed_label','unmatched_detection')),
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),
  import_id uuid NULL REFERENCES dataset_imports(id) ON DELETE CASCADE,
  body jsonb NOT NULL,                       -- pydantic note body: detector class, IoU, counts
  created_at timestamptz NOT NULL DEFAULT now()
);  -- wire proposal_chain = the item's 'proposal' rows in note_key order
CREATE TABLE faces (                         -- one detected face; replaces visual_search_faces under a project
  face_key bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,
  photo_key bigint NOT NULL REFERENCES photos(photo_key) ON DELETE CASCADE,
  face_id uuid NOT NULL,                     -- content-derived from image_id and the rounded box
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),
  person_item_key bigint NULL REFERENCES items(item_key) ON DELETE SET NULL,   -- person_detection_id
  bbox real[] NOT NULL, landmarks real[] NOT NULL CHECK (cardinality(landmarks) = 10),
  det_score real NOT NULL, quality real NULL, yaw_deg real NULL, eyes_open boolean NULL, side_px integer NULL,
  UNIQUE (project_id, face_id)
);
CREATE TABLE vector_gaps (                   -- why a subject has no vector in a space: the reader's explanation, never a work list
  space_id uuid NOT NULL REFERENCES embedding_spaces(id) ON DELETE CASCADE, PROJECT,
  subject_key bigint NOT NULL,
  reason text NOT NULL CHECK (reason IN ('not_selected','failed')),   -- src/domain/embedding_state.py:19-26
  PRIMARY KEY (space_id, subject_key)
);  -- a vector row's existence is "embedded"; "deferred" is the ledger's 'pending' row of 'encoder:<space_key>' (m7);
    -- the vector writer deletes the gap in the same transaction
```

Changes to 02d tables, in the same revision: `cluster_memberships.item_key text` becomes `subject_key
bigint NOT NULL` (an `item_key`, `box_key`, `face_key` or `photo_key` by the partition's `target`),
primary key `(partition_id, subject_key)`, no foreign key (02d 2.2's immutability trigger would refuse
the cascade; erasure deletes under its guard, section 8). It is the only placement store: cluster
reads join the serving partition (`cluster_activations.serving_partition_id`), so an activation
rewrites no item row (M6). `near_dup_members.member_id`, `near_dup_edges.a_id`, `b_id` and
`holdout_near_dup_moves.item_id` become `bigint` keys the same way. Duplicate-group reads (M19): item
or photo to group through the members' key `(run_id, purpose, member_id)` of the current run, group
to members through `ix_near_dup_members_group` ([09d](09d_research_near_duplicates.md)).

Sizes at 10 M (heaps at fillfactor 100):

| Table | Tuple arithmetic | Heap |
|---|---|---|
| `items` | 24 + 8 + 16 + 8 + 16 + 3 x 4 + 8 = 92, +4 → 96; 10 M x 96 / 8,168 | **1.2 GB** |
| `detections` | 24 + 8 + 16 + 8 + 8 + 4 + 40 + 8 + 4 + 4 + 8 + 12 (+ NULLs) ≈ 150; 10 M x 154 / 8,168 | **1.9 GB** |
| `class_opinions` | about 1 per item: 24 + 8 + 16 + 12 + 4 + 12 + 4 + 8 + 4 + 8 + 12 + 8 ≈ 120; 10 M x 124 / 8,168 | **1.5 GB** |
| `item_region_results` | 10 M x about 110 / 8,168 | **1.4 GB** |
| `region_boxes` | 5 M x about 134 / 8,168 | **0.8 GB** |
| `analysis_ledger` | 3 M photos x 8 kinds x 64 / 8,168 | **1.5 GB** |
| `photo_capture`, `text_lines`, `probe_predictions`, `item_scores`, `item_projections`, `assist_answers`, `item_notes`, `faces`, `vector_gaps` | filled by later or optional analysers; each below 1.5 GB at 10 M (`item_scores` 20 M x 60 B = 1.2 GB the largest) | ≤ **5 GB together** |

### 2.4 Current state: the fold's output

`item_state` holds only what changes through a claim, the effective class, or the region status
(M5). Machine scores (probe, uniqueness, mistakenness, projection coordinates, blur, detector
values) stay in their result tables and are joined; capture time is the photo's; cluster placement
is `cluster_memberships` (section 2.3).

```sql
CREATE TABLE item_state (                    -- one narrow row per item (plan 15 section 4.4)
  item_key bigint PRIMARY KEY REFERENCES items(item_key) ON DELETE CASCADE, PROJECT,
  photo_key bigint NOT NULL, crop_id uuid NOT NULL,   -- copies: a page needs no join for ids; crop_id is today's sort tiebreak (list_cursor.py)
  bbox real[] NOT NULL, area real NOT NULL, rank smallint NOT NULL,   -- effective geometry (4.3 layer 1), crop_area_norm, crop_rank_in_image
  geometry_source text NOT NULL, geometry_revision integer NOT NULL,   -- 'detector' | 'human' | 'import'
  class_id integer NULL, class_source text NULL, label_source text NULL,   -- effective class (layer 2); the name is joined, never copied
  class_validated boolean NOT NULL DEFAULT false, class_unresolved boolean NOT NULL DEFAULT false,
  class_labeled_at timestamptz NULL, class_confidence real NULL,      -- confidence of the deciding opinion or detection
  class_claim_id bigint NULL, class_op_id uuid NULL, class_opinion_source text NULL,   -- what decided the class (label_op_id, provenance)
  vlm_suggestion_dismissed boolean NOT NULL DEFAULT false, needs_new_class boolean NOT NULL DEFAULT false,
  probe_dismissed_class text NULL,           -- the class a live probe_dismissed named; "dismissed" = it equals the current prediction (read)
  excluded boolean NOT NULL DEFAULT false, excluded_reason text NULL, excluded_at timestamptz NULL,
  holdout boolean NOT NULL DEFAULT false, review_dismissed_at timestamptz NULL,
  audit_sample boolean NOT NULL DEFAULT false, audit_outcome text NULL,
  combine_conflict boolean NOT NULL DEFAULT false, import_ids uuid[] NULL,
  region_status text NULL, region_validated boolean NOT NULL DEFAULT false, region_verifier text NULL, region_label_source text NULL,
  region_count smallint NOT NULL DEFAULT 0, region_rejected_count smallint NOT NULL DEFAULT 0, region_max_score real NULL,
  region_revision integer NOT NULL,          -- the editor's OCC token (expected_region_revision)
  region_set_complete boolean NULL,          -- fold output (Q8): the closed set (4.3 layer 4); NULL while no box claim or machine set decides
  region_seen_seq integer NOT NULL DEFAULT 0,   -- fold output (Q9): the machine-box generation the fold shows; the claim writer stores it as a claim's region_seq
  geometry_pinned boolean NOT NULL DEFAULT false,   -- fold output (Q11): bbox is the box a person saw, exactly while today's is_locked_item holds
  moved_partition_id uuid NULL, moved_cluster_uid uuid NULL,   -- a live class_moved (layer 5): read only while that partition serves
  created_at timestamptz NOT NULL, updated_at timestamptz NOT NULL   -- created_at: copy of items.created_at (15b section 7 sort path); updated_at: see the write rules below
);  -- no UNIQUE (project_id, crop_id): items holds it, the copy here is for sort order only (m3)
-- 0011 adds fold outputs that summary keys and review tabs count (15b 6.2), not indexed:
--   detector_class_name text NULL, detector_confidence real NULL   (the detection used for machine geometry, 3.4 step 6)
--   vlm_confidence text NULL, vlm_class_empty boolean NOT NULL DEFAULT false   (the live VLM opinion)
CREATE INDEX ix_item_state_class_op ON item_state (class_op_id) WHERE class_op_id IS NOT NULL;
  -- shape 46 (M8): an operation's current count is one index range count; class_op_id changes only with the deciding claim
ALTER TABLE item_state SET (autovacuum_vacuum_scale_factor = 0.01, autovacuum_vacuum_threshold = 1000,
                            autovacuum_vacuum_cost_limit = 2000);
CREATE TABLE box_state (                     -- one row per live box: machine box folded with box claims
  box_row_key bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,
  item_key bigint NOT NULL REFERENCES items(item_key) ON DELETE CASCADE,
  box_id text NOT NULL,
  box_key bigint NULL REFERENCES region_boxes(box_key) ON DELETE SET NULL,   -- NULL: a box that exists only through a claim
  bbox real[] NOT NULL, state text NOT NULL, score real NULL, detector text NULL, detector_version text NULL,
  source text NULL, bbox_correct boolean NULL, confidence text NULL, rejection_reason text NULL,
  text text NULL, text_source text NULL, text_choice text NULL, text_disagreement boolean NULL,
  text_fold text NULL,                       -- today's op_text_compact normalisation, written by one domain function
  ordinal smallint NOT NULL, detected_at timestamptz NULL,
  UNIQUE (item_key, box_id)
);
ALTER TABLE box_state SET (autovacuum_vacuum_scale_factor = 0.01, autovacuum_vacuum_threshold = 1000,
                           autovacuum_vacuum_cost_limit = 2000);
ALTER TABLE claims SET (autovacuum_vacuum_insert_scale_factor = 0.01);   -- append only: keeps the visibility map set
```

Write rules (M5): the fold writes `UPDATE ... WHERE item_key = :k AND (cols) IS DISTINCT FROM (new)`,
so a no-op writes no tuple. `updated_at` moves only when a fact a person sees on the item changes
(class, validation, exclusion, holdout, geometry, review dismissal, a person's region or box
verdict), not for region-worker, VLM-opinion or count-only columns, so those writes touch no indexed
column and can stay HOT, and machine churn does not reorder "Recently updated" (review of 15b m6;
owner confirmation: 15b section 11 question 8). Every state write goes through
`state.store_states`, which also emits the summary deltas (15b 6.2). An update is HOT when "no
columns referenced by the table's indexes" change and the page has room
([HOT](https://www.postgresql.org/docs/18/storage-hot.html)): the target, checked in the gate A sweep
from `pg_stat_user_tables.n_tup_hot_upd / n_tup_upd`, is at most 5 B-trees on `item_state` and a
label write HOT whenever no indexed column changes. The default autovacuum (scale factor 0.2, cost
limit 200, delay 2 ms, [autovacuum](https://www.postgresql.org/docs/18/runtime-config-vacuum.html))
would wait for 2 M dead rows on 10 M; the settings above start at 1 % with ten times the work per
run. Both tables are derived (R7): `fold.rebuild` refolds them in chunks (4.1); `items` is never
emptied.

Sizes at 10 M: `item_state` tuple ≈ 24 header + 16 project + 3 x 8 keys + 16 crop id + 16 op id + 40
bbox + 28 fixed numbers + 10 booleans + 24 (three timestamps) + about 70 of short text + 6 null
bitmap ≈ 290 bytes; 10 M x 294 / 8,168 = 360 k pages, **2.9 GB** at fillfactor 100. `box_state`: 5 M
x 140 / 8,168 = **0.9 GB**.

### 2.5 Gallery: entities, edges, photo state (phase 5; designed now, created now, empty until then)

Identity rows (`entities`, like `items`) never change; names, hierarchy and album fields are claims
folded into `entity_state`. `revision` on `entity_state` and `photo_state` is the `expected_revision`
of every library write (photo_library 00 section 7): the subject's claims and withdraws counted, so it
only grows and a rebuild reproduces it.

```sql
CREATE TABLE entities (                      -- people, places, events, trips, albums, keywords, memories
  entity_key  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,
  entity_id   uuid NOT NULL DEFAULT uuidv7(),   -- wire id (persons.id, album_id, keyword_id, memory_id of photo_library 00 section 6.1, 30)
  kind        text NOT NULL CONSTRAINT ck_entities_kind CHECK (kind IN ('person','place','event','trip','album','keyword','memory')),
  analyser_id integer NULL REFERENCES analysers(analyser_id),   -- set when a machine made it (grouping, events, places, memories)
  cluster_uid uuid NULL REFERENCES cluster_identities(cluster_uid) ON DELETE SET NULL,  -- a person's face cluster identity
  geonames_id integer NULL,                   -- a place: geo_places row (photo_library 10)
  created_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (project_id, entity_id)
);  -- a person made by a person (split), an album or a user keyword is created by an 'entity_created' claim
CREATE TABLE entity_edges (                  -- machine links of a subject to an entity: results
  entity_key  bigint NOT NULL REFERENCES entities(entity_key) ON DELETE CASCADE, PROJECT,
  subject_kind text NOT NULL CHECK (subject_kind IN ('photo','face','item')),
  subject_key bigint NOT NULL,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),
  weight      real NULL,
  PRIMARY KEY (entity_key, subject_kind, subject_key)
);
CREATE TABLE entity_state (                  -- fold of entity claims (layer: entity)
  entity_key bigint PRIMARY KEY REFERENCES entities(entity_key) ON DELETE CASCADE, PROJECT,
  kind text NOT NULL,                        -- copy of entities.kind, for the keyword key below
  name text NULL, name_key text NULL,        -- album title, person or keyword name; name_key = normalised name
  parent_key bigint NULL REFERENCES entities(entity_key) ON DELETE CASCADE,   -- keyword hierarchy
  path text NULL,                            -- materialised 'a/b/c' of a keyword ("with descendants" = path prefix)
  description text NULL, cover_photo_key bigint NULL REFERENCES photos(photo_key) ON DELETE SET NULL,
  sort text NULL CHECK (sort IN ('position','taken_asc','taken_desc')),   -- album fields (photo_library 00 albums)
  hidden boolean NOT NULL DEFAULT false, merged_into bigint NULL, state text NOT NULL,
  member_count integer NOT NULL DEFAULT 0, revision bigint NOT NULL, updated_at timestamptz NOT NULL
);
CREATE UNIQUE INDEX uq_entity_state_keyword ON entity_state (project_id, parent_key, name_key) NULLS NOT DISTINCT
       WHERE kind = 'keyword';               -- photo_library 00 keywords: unique (project, parent, name); a clashing rename is 409
CREATE TABLE membership_state (              -- fold of entity_edges and membership claims (albums, keywords, memories, person confirm/reject)
  entity_key bigint NOT NULL REFERENCES entities(entity_key) ON DELETE CASCADE, PROJECT,
  subject_kind text NOT NULL, subject_key bigint NOT NULL,
  source text NOT NULL CHECK (source IN ('machine','human')), position numeric NULL, weight real NULL,
  PRIMARY KEY (entity_key, subject_kind, subject_key)
);
CREATE TABLE face_state (                    -- fold: the face's person (grouping result + confirm/reject claims)
  face_key bigint PRIMARY KEY REFERENCES faces(face_key) ON DELETE CASCADE, PROJECT,
  person_entity_key bigint NULL REFERENCES entities(entity_key) ON DELETE SET NULL,
  person_state text NOT NULL CHECK (person_state IN ('unassigned','suggested','assigned','confirmed')),
  assign_score real NULL, assign_source text NULL
);
CREATE TABLE photo_state (                   -- fold of photo and media claims over photo_capture: what the timeline and map read
  photo_key bigint PRIMARY KEY REFERENCES photos(photo_key) ON DELETE CASCADE, PROJECT,
  taken_local timestamp NULL, taken_utc timestamp NULL, time_source text NULL,   -- effective: correction claim, else read
  lat double precision NULL, lon double precision NULL, location_source text NULL,
  favourite boolean NOT NULL DEFAULT false, pick text NOT NULL DEFAULT 'none', hidden boolean NOT NULL DEFAULT false,
  archived boolean NOT NULL DEFAULT false, trashed_at timestamptz NULL,
  title text NULL, caption text NULL, stack_pick_group uuid NULL,   -- a human top pick of a stack (photo_library 20 stack_picks)
  revision bigint NOT NULL, updated_at timestamptz NOT NULL
);
CREATE TABLE photo_files (                   -- added data: the other files of one photo (photo_library 05 section 2.5)
  photo_key bigint NOT NULL REFERENCES photos(photo_key) ON DELETE CASCADE, PROJECT,
  role text NOT NULL CHECK (role IN ('motion','edited','recipe','raw','sidecar')),
  storage_key text NOT NULL, size_bytes bigint NOT NULL, content_sha256 bytea NULL,
  source_state text NOT NULL DEFAULT 'present' CHECK (source_state IN ('present','moved','missing')),
  PRIMARY KEY (photo_key, role, storage_key)
);
CREATE TABLE photo_derivatives (photo_key bigint PRIMARY KEY REFERENCES photos(photo_key) ON DELETE CASCADE, PROJECT,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),
  thumbhash bytea NULL, aspect real NOT NULL, grid_key text NULL, at timestamptz NOT NULL DEFAULT now());   -- N3
CREATE TABLE photo_signals (photo_key bigint PRIMARY KEY REFERENCES photos(photo_key) ON DELETE CASCADE, PROJECT,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),
  sharpness real NULL, exposure real NULL, aesthetic real NULL, utility_class text NULL);                     -- N6
CREATE TABLE photo_places (photo_key bigint PRIMARY KEY REFERENCES photos(photo_key) ON DELETE CASCADE, PROJECT,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),   -- reverse geocoder + geo_places dataset version
  geonames_id integer NULL, country_code text NULL, admin1 text NULL, admin2 text NULL, locality text NULL);
  -- a result of the effective position (photo_state.lat/lon): recomputed when a position claim changes it
```

These replace the mutable `photo_annotations`, `capture_overrides`, `media_text`, `keywords`,
`media_keywords`, `albums`, `album_members`, `persons`, `face_constraints`,
`library_operation_members`, `stack_picks` and `exclusion_photo_decisions` of
[photo_library 00](../photo_library/00_photo_library_plan.md) section 6.1 and
[20](../photo_library/20_p2b_curate.md) section 5 (plan 15 R3; section 14, answer 3). Memories
([30](../photo_library/30_p2c_create.md) section 6) are entities of kind `memory`: generator picks are
`entity_edges`, a person's edits are claims (3.2). Exclusion rules, smart albums, media tracks and
`geo_places` stay as those plans design them; the plans are updated to these tables before phase 5.
Sizes: empty in phase 2; at 3 M photos `photo_state` ≈ 3 M x 130 / 8,168 = 0.5 GB.

### 2.6 From gate A: indexes, summary tables, fillfactors

Filled 2026-11-15 from the gate A measurement, revised the same day by the review of 15b and 15c
(appendix A, "Reads and slices review") (`artifacts_local/speed_1113/v2_read_shapes/`,
`notes.md` section 3 for the spike's index set) and the read map
[15b_storage_reads.md](15b_storage_reads.md), which gives the query each index serves (its section
3, shape numbers of the storage inventory). Rules applied: one composite index per read in the
read's order (equality columns, the sort column in its direction, the tiebreak last); keyset pages
use one row comparison with the tiebreak in the sort's direction (15b section 4, M7, [row-wise
comparison](https://www.postgresql.org/docs/18/functions-comparisons.html#ROW-WISE-COMPARISON)); the
tiebreak is `crop_id` on `item_state` sorts, `photo_key` on photo sorts and `item_key`
(`subject_key`) on sorts whose key lives in a result or membership table; at most 5 B-trees on
`item_state` besides its key; whole-project counts and aggregates read summary tables kept by the
writer (15b section 6). "Spike" names the gate A index the row comes from (`notes.md` section 3);
"new" rows are to be measured in the first wave sweep after the slice that uses them (15b 1.3).

| # | Table | Columns in order (direction) | Partial predicate | Serves (shapes) | From | Size at 10 M (estimate) |
|---|---|---|---|---|---|---|
| X1 | `item_state` | `project_id, updated_at DESC, crop_id DESC` | | 1, 3-5, 37, 50 (default sort, locate, keyset walks) | spike `o_upd` | 10 M x 52 B / (8,192 x 0.9) ≈ 0.6 GB |
| X2 | `item_state` | `project_id, class_id, class_validated, updated_at DESC, crop_id DESC` | | 1, 3-5 with a class filter (one range per class and validated value; several classes are `UNION ALL` branches, 15b P1), 53, 66 (merge walk) | spike `o_class_upd` (`s1a_deep` 1.3 / 2), validated key from `o_class_flags` (`s1c_deep` 1.4 / 2) | 0.7 GB |
| X3 | `item_state` | `photo_key, rank, crop_id` | | 25, 29, 80 (a photo's items by rank) | spike `o_image` (`s7` 3.1 / 5) | 0.5 GB |
| X4 | `item_state` | `project_id, class_confidence, crop_id` | `NOT class_validated AND NOT excluded AND review_dismissed_at IS NULL` (the rows that need review; it shrinks as the queue drains) | 2 (every queue's capped count and its sort path, `check_machine`), 1 (`confidence` sort on review-predicate lists) | spike `o_review_conf`, re-keyed: new | at most 7 M rows x 52 B ≈ 0.4 GB |
| X5 | `item_state` | `class_op_id` | `class_op_id IS NOT NULL` | 46 (section 2.4, `0010`) | `0010`, kept | ≈ 0.1 GB |
| X6 | `box_state` | `project_id, state, detector, item_key DESC` | | 6 (box rows, 15b section 9), 52 | new | 5 M x 60 B ≈ 0.3 GB |
| X7 | `box_state` | GIN `text_fold gin_trgm_ops`, `fastupdate = on`, `gin_pending_list_limit = 512` | `text_fold IS NOT NULL` | fragment predicates of 2, 6 and 1, through `op_text_fragment` (15b section 8: row security keeps `LIKE` off the index otherwise) | round 3 M9 (108 MB at 10 M) | ≈ 0.1 GB |
| X8 | `detections` | `project_id, analyser_id, confidence DESC NULLS LAST, item_key DESC` | | `detector_confidence` sort, `to_label` default (15b section 7) | new | 10 M x 48 B ≈ 0.55 GB |
| X9 | `photos` | `project_id, photo_key` | | 24 (`indexed_at` sorts as insert order), 51, 55, 58 and every photo walk (`analysers.photos_needing`) | new | 3 M x 36 B ≈ 0.15 GB |
| X10 | `photos` | BRIN `created_at` | | 31d (ingest status, 14 days), 79 (new photos since) | new ([BRIN](https://www.postgresql.org/docs/18/brin-intro.html): insert-ordered time) | < 1 MB |
| X11 | `photos` | `project_id, source_identifier` | `source_identifier IS NOT NULL` | 31a | 2.6 revision 0 candidate | small |
| X12 | `photos` | `project_id, storage_key, photo_key` | | 31b, 31c, 24 (`file_name` sort) | new | 3 M x 100 B ≈ 0.4 GB |
| X13 | `photo_capture` | `project_id, taken_local DESC NULLS LAST, photo_key DESC` | | 24 (`taken_at` sorts, today on `capture.taken_local`, `image_view.py:65`) | spike `i_taken` (`s7_deep` 3.7 / 6) | 0.15 GB |
| X14 | `photo_capture` | GiST `point(lon, lat)` | `lat IS NOT NULL` | 30, 24 (geo filter) | spike `i_loc` (`s11_map` 18.0 / 171) | ≈ 0.05 GB |
| X15 | `item_region_results` | `project_id, item_key` | `status IN ('pending_detection', 'pending_verification')` | 49 (worker poll) | new | transient, small |
| X16 | `analysis_ledger` | `project_id, kind, photo_key` | `state <> 'done'` | 55, 79, failed and pending re-runs (P14); the full work list is `analysers.photos_needing` over X9 and the ledger key | new | small |
| X17 | `class_opinions` | `op_id` | `op_id IS NOT NULL` | undo of an adoption or a propagation (3.5) | 2.6 revision 0 candidate | small |
| X18 | `class_opinions` | `project_id, raw_label` | `raw_label IS NOT NULL` | 18 | 2.6 revision 0 candidate | ≈ 0.2 GB |
| X19 | `class_opinions` | `project_id, proposed_class` | `outcome = 'new_class_pending'` | 17, 19 | new | small |
| X20 | `item_scores` | `project_id, kind, score DESC, item_key DESC` | | `mistakenness_score`, `uniqueness_score` sorts | new | 20 M x 44 B ≈ 1.1 GB |
| X21 | `probe_predictions` | `project_id, entropy, item_key` | | 2 (`disagreements` default), 56 | new | 0.5 GB |
| X22 | `cluster_memberships` | `partition_id, cluster_uid, score DESC NULLS LAST, subject_key DESC` (one direction: a backward scan serves the ascending form, 15b P8) | | 1 cluster filter, 10 rebuild, 12, 13, 41, 77, 6 inside a box cluster; replaces 02d `ix_cluster_memberships_cluster` | spike `pl_cluster` (`pg_pl` pending) | per retained partition 10 M x 44 B ≈ 0.5 GB |
| X23 | `cluster_memberships` | `partition_id, score, subject_key` | | 2 (cluster-fit sorts), 8 (`core_first`) | spike `pl_score` (`pg_pl` pending) | ≈ 0.45 GB per partition |
| X24 | `photo_counts` | `project_id, unlabeled DESC, photo_key DESC` | | 24 (`unlabeled_count` sorts, `label_state` filter) | new | 0.15 GB |
| X25 | `text_lines` | GIN `lower(text) gin_trgm_ops`, `fastupdate = on`, `gin_pending_list_limit = 512` | `item_key IS NOT NULL` | item text predicate of 1 and 2, through `op_text_word` (15b section 8) | new | small |
| X26 | `box_state` | `project_id, state, item_key DESC` | | 6 with a state and no detector | new (review M6) | 0.25 GB |
| X27 | `box_state` | `project_id, detector, item_key DESC` | | 6 with a detector and no state | new (review M6) | 0.25 GB |
| X28 | `items` | `project_id, created_at DESC, crop_id DESC` | | `created_at` sorts | new (review M3) | 0.6 GB |
| X29 | `items` | `project_id, sort_area, crop_id` | | `crop_area_norm` sorts | new (review M3) | 0.5 GB |
| X30 | `items` | `project_id, sort_rank, crop_id` | | `crop_rank_in_image` sorts | new (review M3) | 0.45 GB |
| X31 | `detections` | `project_id, analyser_id, blur_lap_ratio, item_key` | | `blur_lap_ratio` sorts | new (review M3) | 0.5 GB |
| X32 | `detections` | `project_id, analyser_id, class_name, item_key` | | `detector_class_name` sorts; the detector-class filter and its capped count (`EXISTS` on `detections`, 15b P1) | new (review M3) | 0.55 GB |
| X33 | `box_state` | `project_id, text_fold` | `text_fold IS NOT NULL` | exact, fuzzy (the distance-1 variant list) and same-text predicates of 2, 6 and 1 (15b section 8, revision 2) | round 3 M9: equality on X7 under row security 64 to 170 ms for variant lists | 5 M x 40 B ≈ 0.2 GB |

**`item_state` holds exactly five B-trees besides its key** (X1 to X5). Sort values a label never
changes live on `items` and `detections` and are indexed there (X8, X28 to X32), where a label
writes nothing (review of 15b M3; 15b section 7). The foreign-key index
`ix_item_state_project_id` that `0010` created is dropped: X1 and X2 lead with `project_id` and are
not partial, so they serve the lookup a project delete's cascade needs ("it is often a good idea to
index the referencing columns", [foreign keys](https://www.postgresql.org/docs/18/ddl-constraints.html#DDL-CONSTRAINTS-FK)),
and the schema test asserts the replacement instead of the plain index. No sixth index: the spike's
`o_class_flags` is X2 with `class_validated` as a key (excluded and holdout checked on the heap; the
end-game queue is the partial X4); the spike's
`o_cluster_*` and `o_review` move to the membership table (X22, X23) with the placement; its
`o_cluster_cov` and `o_cls_src` are replaced by the summary tables. `updated_at` is in X1 and X2, so
every person-visible change of a state row is a non-HOT update and inserts one entry into each
index ([HOT](https://www.postgresql.org/docs/18/storage-hot.html) needs "no indexed column
modified"): the price of the default "Recently updated" order; machine-only changes do not move
`updated_at` (2.4) and can stay HOT. Measured before `0011` is written (15b section 13 item M5).

**Summary tables**: base rows, the insert-only delta log `sum_deltas`, the `_now` views, the
compactor and the rebuild are in 15b section 6. `sum_items`, `sum_review`, `sum_region`, `sum_box`,
`sum_photos`, `photo_counts`, `sum_deltas` and their views are created by `0011`; `sum_cluster`,
`sum_cluster_class`, `sum_cluster_review`, `sum_cluster_dup`, the card rows `sum_cluster_card` (15b
revision 2) with X22 and X23 by
`0016_cluster_cutover`, because they need `cluster_memberships.subject_key` (section 2.3). Writers
only insert delta rows, so no summary row and no advisory lock enters the lock order of section 7.

**Planner statistics**: `CREATE STATISTICS st_item_state_class (dependencies, mcv) ON class_id,
class_source, class_validated, excluded, holdout FROM item_state` (the flags follow the class source)
and `ALTER TABLE item_state ALTER COLUMN class_id SET STATISTICS 1000` (a registry of up to 1,000
classes in the most-common-values list) ([CREATE STATISTICS](https://www.postgresql.org/docs/18/sql-createstatistics.html),
[ALTER TABLE](https://www.postgresql.org/docs/18/sql-altertable.html)).

| Table | fillfactor | autovacuum | Reason |
|---|---|---|---|
| `item_state` | 90 (15b section 14.2: region-only updates 100 % HOT at 90, 6.2 % at 100; 80 if round 4 R4-7 shows under 90 % HOT beside labels) | `0010`'s: scale factor 0.01, threshold 1,000, cost limit 2,000 | a person-visible change moves `updated_at` (X1, X2) and is never HOT; machine-only changes can be HOT (2.4) only with free space on the page, so S1a-i sets the fillfactor from M5's HOT share; dead versions arrive at label rate, hence vacuum at 1 % with ten times the default work ([autovacuum](https://www.postgresql.org/docs/18/runtime-config-vacuum.html)) |
| `box_state` | 90 | `0010`'s | a region re-run that moves a box or changes its score without a state, detector or text change touches no indexed column and can stay HOT |
| `class_opinions`, `item_region_results`, `probe_predictions`, `item_scores`, `item_projections`, `assist_answers` | 90 | default | a re-run replaces values that no index holds (status leaves X15 only when pending ends) |
| `analysis_ledger` | 90 | default | `updated_at`, `reason`, `analyser_id` change on re-runs; `state` leaves X16 when done |
| `items` | 90 | default | `box_seq`, `geometry_seq`, `region_seq` are not indexed (section 2.3): HOT |
| `sum_*` base tables | 50 | `autovacuum_vacuum_scale_factor = 0`, `autovacuum_vacuum_threshold = 2000` | small rows updated only by the compactor, counter not indexed: HOT needs room on the page (the spike's 50); a percentage of a tiny table would never trigger, so vacuum by count |
| `sum_deltas` | 100 | scale factor 0, threshold = the compactor batch (15b 6.1, section 13 M3) | an insert-then-delete queue |
| `photo_counts` | 90 | default | `unlabeled` is in X24, so a label is not HOT here; 90 keeps the new version on the page |
| `claims`, `photos`, `detections`, `text_lines`, vector tables | 100 | `claims` insert scale factor 0.01 (`0010`) | insert-mostly |

**Migration `0011_storage_indexes`** (slice S1a-i, 15c; Rust slice R1b): `CREATE EXTENSION IF NOT
EXISTS pg_trgm`; X1 to X21, X24 to X33; the role `op_text_search` and the functions
`op_bound_project_strict`, `op_text_fragment`, `op_text_word` (15b section 8); the drop of
`ix_item_state_project_id`; the `items` and `item_state` columns of 2.3 and 2.4; the summary tables,
`sum_deltas` and views above with RLS(02a); the statistics object and the column target; the
fillfactor and autovacuum settings. The tables are empty when it runs (the
data is wiped and re-ingested after the cut-over), so the indexes are built in the migration's
transaction; an index added later on a filled table uses `CREATE INDEX CONCURRENTLY` in an
autocommit block, which "cannot be executed inside a transaction block"
([CREATE INDEX](https://www.postgresql.org/docs/18/sql-createindex.html)). With this revision the
cut-over revision ids of section 9 move: `0012_ingest_cutover` (S1a-ii: `image_contents` keeps
only the core API's rows, `project_id IS NULL`, fenced by a `CHECK`; the table goes with the core
routes in phase 3), `0013_claims_cutover` (S2a), `0014_vector_cutover` (S3: the `embedding_spaces` change of 6.3),
`0015_near_dup_cutover` (S5), `0016_cluster_cutover` (S6), `0017_import_cutover` (S7),
`0018_reads_cutover` (S9: drops `project_indexes` and `search_projection_tasks`); 15c section 1
owns that list.

The vector reads (20-23, 59-65) are gate B's (section 6.4).

## 3. The claim log

### 3.1 Columns

DDL in section 2.2. A claim names: when (`at`, and `claim_id` for order), who (`actor_kind`,
`actor`), the operation it belongs to (`op_id`), what (`kind`, `payload`) and the subject.

### 3.2 Kinds (the rows of `claim_kinds`)

One list in `src/domain/claim_kinds.py`, equal to the `claim_kinds` rows by test. Pins: R3's pin
(4.3 layer 1). Asserts: an unmatched claim creates an item (3.4 step 3); otherwise it stays dormant.

| Kind | Subject | Layer | Pins | Asserts | Today's route or writer (decorator file:line) |
|---|---|---|---|---|---|
| `item_drawn` | item | geometry | yes | yes | `POST /images/{image_id}/items` (`src/routers/curation/image_items.py:21`); an import creating an item (`dataset_import/item_labels.py:179`) |
| `geometry_set` | item | geometry | yes | yes | `PATCH /crops/{crop_id}/geometry` (`crops_geometry.py:39`) |
| `class_set` | item | class | yes | yes | `PUT /crops/{crop_id}/label` (`crops_write.py:39`), `PUT /crops/batch_label` (`:108`), `POST /label_ops/accept_machine` (`label_ops.py:151`), `POST /review/new_class_proposals/resolve` (`review_resolve.py:111`), validated import labels |
| `class_cleared` | item | class | no | no | `DELETE /crops/{crop_id}/label` with no human label on record, `POST /crops/{crop_id}/discard` with `clear_class` (`label_undo.py:109,:131`) |
| `class_moved` | item | class | no | no | `POST /crops/move` (`crops_write.py:161`); also gives placement (layer 5) |
| `holdout_frozen` | item | class | yes | yes | `POST /test_holdout/freeze` (`holdout_freeze.py:19`); import and combine freezes |
| `excluded`, `restored` | item | flag | `excluded` yes, `restored` no (Q11: the fold pins only a human delete, `not_an_object`) | no | `POST /crops/batch_exclude`, `/crops/batch_unexclude` (`crops_exclude.py:29,:68`) |
| `review_dismissed`, `review_restored` | item | flag | no | no | `/review_dismiss` (`crops_exclude.py:95`), `/discard` with `dismiss_from_review`, `/review_undismiss` (`label_undo.py:228`) |
| `vlm_dismissed`, `probe_dismissed`, `needs_new_class` | item | flag | no | no | `/vlm_dismiss` (`label_undo.py:180`), `/probe_dismiss` (`:197`), `POST /crops/flag_new_class` (`crops_write.py:216`) |
| `audit_sampled`, `combine_conflict` | item | flag | no | no | `POST /audit/start` (`audit.py:23`); the combine job (`projects/combine/*`) |
| `region_status_set` | item | region | yes | no | only the verdict over an empty box list (B3, Q7): `no_region_visible`, or the status the person named (`verify_rejected`, `false_positive`) kept as said: `PATCH /crops/{crop_id}/region_meta` (`regions_edit.py:118`), `POST /regions/batch_status` (`:272`); every other whole-set status is expanded into box claims (3.3) |
| `box_created`, `box_moved`, `box_state_set`, `box_text_set` | box | box | yes | yes | `PUT /crops/{crop_id}/regions` (`regions_boxes_edit.py:291`, as a diff), `PUT /crops/batch_regions` (`:335`), `PATCH /crops/{crop_id}/regions/{box_id}` (`:368`), `POST /regions/batch_box_state` (`:426`), the whole-set routes above; validated import boxes |
| `box_deleted` | box | box | yes | no | the same routes; never creates a box |
| `duplicate_verdict` | pair | pair | no | no | `POST /near_duplicates/decisions` (`near_duplicates.py:110`); the `DELETE` is a `withdraw` |
| `withdraw` | claim | withdraw | no | no | every undo route: `/label/undo`, `/label/undo_batch`, `/label_ops/{op_id}/undo`, region, geometry, dismiss undos (`label_undo.py:62,:78`, `label_ops.py:49`, `crops_geometry.py:64`, `edit_undo.py:156,:175`), import undo, `forget_holdout_freeze` |
| phase 5: `favourite_set`, `pick_set`, `hidden_set`, `archived_set`, `trashed`, `untrashed`, `stack_pick_set`, `exclusion_forced`, `exclusion_released` | photo | photo | no | no | the library routes of photo_library 00 section 7 and 20 section 6 |
| phase 5: `capture_corrected`, `text_set` | media | photo | no | no | the same; a video subject uses `subject_video_key` |
| phase 5: `entity_created`, `entity_named`, `entity_merged`, `entity_hidden`, `member_added`, `member_removed`, `memory_state_set`, `memory_timeline_set` | entity | entity | no | no | the same; `PUT /persons/{person_id}/name`, `POST /persons/merge` (`src/routers/persons.py`) for the core people |
| phase 5: `face_confirmed`, `face_rejected` | face | face | no | no | the people routes of photo_library 15 section 3 |

**What pins the item's box (Q11, `claim_kinds.pins_geometry`).** The pin kinds above say which
claims *can* pin; the fold pins the item's box to the one the person saw exactly while today's
`is_locked_item` (`occ_locks.py:137`) holds: a person's or validated-import class, a holdout, human
or import geometry, a human delete (`excluded` with reason `not_an_object`, which today keeps the
box so a re-detect cannot re-create it), any human-owned box inside the item, or a box set validated
by a person or an import (`ItemState.geometry_pinned`). Not pinning, as today: a `class_cleared`, the
dismissals, the flags, a `class_moved`, an `excluded` with another reason. A person's box was drawn
against the crop they saw, so the item's box must not move under it. The truth table asserts
`geometry_pinned` equals today's `is_locked_item` on every enumerated history.

Class registry edits (create, rename, deprecate, merge, hotkeys) stay rows of `project_classes` with
their revision (02a section 3): they are configuration, not a decision about an object. Adoption of
the detector's class and propagation are not claims: they write not-rebuildable opinion rows
(section 2.3) whose operation's undo deletes them.

### 3.3 Payloads (pydantic models in `src/domain/claim_payloads.py`, `extra='forbid'`)

| Kind | Payload | Note |
|---|---|---|
| `item_drawn` | `{bbox, request_id?, standalone_region?: bool, origin?: {project, crop_id, image_id}}` | `subject_crop_id` = `drawn_crop_id(image_id, request_id)`; an import's item uses the detector-style `crop_id` |
| `geometry_set` | `{bbox}` | `subject_machine_bbox` = the detector box at the time (today `detector_bbox_norm`) |
| `class_set` | `{class_name, class_id_hint, label_source, via: 'single'\|'batch'\|'bulk_accept'\|'resolve'\|'import'\|'combine', labeler, profile: {detector?, detector_version?}}` | `label_source` is one of `HUMAN_LABEL_SOURCES`, `human_bulk_accept` or `import`; `labeler` and `profile` are the wire `class_labeler`, `class_detector`, `class_detector_version` at write time (`class_label.py:45`, M19) |
| `class_cleared` | `{hidden: [{source, analyser_id}]}` | Q10: every machine opinion and detection present when the person cleared, as (source, analyser) pairs (`source` `detection` for a detection); an opinion is hidden iff its pair is in the list of a live clear, so a re-run of the same analysers leaves the item cleared and a newer model version shows |
| `excluded`, `restored`, `review_*`, `needs_new_class` | `{reason}`, `{}`, `{}`, `{note?}` | `excluded.reason` `not_an_object` is a delete; `restored` is the one way back from `excluded` (M2) |
| `class_moved` | `{partition_id, cluster_uid}` | |
| `vlm_dismissed` | `{class_name?, class_id_hint?}` | the suggestion that was rejected |
| `probe_dismissed` | `{class_name, probe_analyser_id}` | |
| `holdout_frozen` | `{freeze_id, machine_class: {class_name, source, confidence} \| null}` | required; `null` = frozen as "no class" (M1): the machine class that would win at the freeze, or none (4.3 layer 2) |
| `audit_sampled` | `{batch_id, label_name, label_source}` | |
| `region_status_set` | `{status: 'no_region_visible'\|'verify_rejected'\|'false_positive', label_source, region_seq, item_bbox}` | Q7: written over an empty box list only; the status is what the person said |
| `box_created` | `{bbox, state, text?, region_seq, item_bbox}` | `subject_box_id` minted from `items.box_seq` |
| `box_moved`, `box_state_set`, `box_text_set`, `box_deleted` | `{bbox, bbox_correct?, region_seq, item_bbox}`, `{state, rejection_reason?, region_seq, item_bbox}`, `{text, region_seq, item_bbox}`, `{region_seq, item_bbox}` | an explicit empty text is a reading of "no text" (`region_boxes.py:277`); `subject_bbox` is the box as shown; `item_bbox` is the item's own box at the write, which the pin (3.2) and matching (3.4 step 3) read |
| `combine_conflict` | `{origins: [..], merged_origins: [..]}` | |
| `duplicate_verdict` | `{level, decision: 'same'\|'different'}` | members in `subject_crop_id`/`photo_key` and `subject_pair_id` |
| `withdraw` | `{}` | the target is `withdraws_claim_id` |
| phase 5 | one model per kind, e.g. `capture_corrected {taken_local?, offset_minutes?, lat?, lon?, altitude_m?, location_source}`, `entity_named {name, parent_entity_id?}`, `stack_pick_set {group_id}`, `exclusion_forced {rule_id}` | the columns of the photo-library tables they replace |

`region_seq` in every box and region claim is `item_state.region_seen_seq` (Q9), the generation of
machine boxes the fold showed the person, not `items.region_seq` (a re-run may have replaced the rows
since; the fold would otherwise show the new rows as the ones the person saw, and a closed set
would look open). The claim writer reads it from the stored fold output under the item lock. A
whole-set status other than `no_region_visible` is expanded at write time, in the same operation,
into what `boxes_with_status` computes (`region_box_edits.py:254-295`): one `box_state_set` for every
box of the resulting list, so a closed set is entirely claims; over an empty list the verdict is one
`region_status_set` carrying the named status (Q7), and `no_region_visible` also writes a
`box_deleted` per box. A partial per-box act writes claims for the boxes it changed only, except
that an edit which closes the set claims every box.

Class in a claim: by name, with the registry id of that moment as a hint (M20). The hint is used only
when `project_classes(project, hint).name_key = normalize_class_name(name)` or the hint's class is
reached from the named class through `merged_into` (then followed to the end); else the active class
of that `name_key` (02a section 3); else the item is `class_unresolved`: locked, no class (4.3 layer
2; fail closed; the claim stays). A hint from another project's registry never selects a class.

### 3.4 Addressing a subject after every result was rebuilt

Photos are never rebuilt, so `photo_key` is as stable as the wire `image_id`. Items and boxes are
found again by `src/domain/claim_match.py`, shared by the fold, re-detect and import (it replaces
`src/services/curation/proposal_merge.py:109-170`), per item, not per claim (M14):

1. **Group** the photo's live claims by `subject_crop_id`. A group whose `subject_frame` differs from
   `photos.frame` matches nothing, stays dormant and is reported `frame_mismatch` (fail closed).
2. **Same id**: an item whose `crop_id` is the group's `subject_crop_id` takes the group.
3. **Else** groups and new detections are matched greedily, best IoU first, ties by `crop_id`: the
   detection box against the group's effective box (layer 1), its newest `subject_machine_bbox`, or
   the `item_bbox` its newest claim saw (Q5: a lone dismissal or delete has no other box to match
   by), at least **0.5** (`LABEL_IOU_MATCH`, `proposal_merge.py:30`). A new matched item takes the group's
   `crop_id`, so a touched item keeps its wire id across detector versions (`reprocess_detect.py:1-11`).
4. **No match**: a group with a live asserting claim (`claim_kinds.asserts_object`) creates an item
   from the claims, `geometry_source` `human` or `import`. A group of dormant kinds only (`excluded`
   `not_an_object`, `review_dismissed`, `class_cleared`, flags) creates nothing and applies to the
   first later matching detection, so a deleted object stays deleted (`occ_locks.py:90-99`).
5. **Item rows**: kept while the item has a detection or a live asserting claim; else `attach`
   deletes the row and its dormant claims wait for step 2 or 3.
6. **Machine geometry** with several detections: the newest `detector` analyser; `open_vocab` only
   without a detector row; `import` only without either.
7. **Boxes**: the same inside the item: a claim on a machine box matches on `subject_box_id`, then
   IoU ≥ 0.5; a box a person created owns its minted id and matches on that id only, never by
   overlap with an unrelated machine box (Q6). A claim group whose id a re-run replaced joins the
   machine box another group took by id when it overlaps it at least 0.5: both name one box, so the
   set never shows a person's old row twice. A split (one claimed
   box, two new boxes each below 0.5) leaves the claimed box and two machine boxes, as today.

### 3.5 Undo, operations, imports, retention

- **Undo** writes one `withdraw` row per claim; there is no operation-level withdraw (B4). A job sets
  the operation `undoing`, walks its claims (`uq_claims_op_subject` leads with `op_id`) in chunks of
  1,000, and in **one transaction per chunk** inserts the withdraws (`ON CONFLICT DO NOTHING`) and
  folds the items, so the state always equals the fold of the live claims; then `undone`,
  `undone_at`. A dead job resumes where the unique index says. Every claim of the operation is
  withdrawn; a dry pass first counts the visible effect per item (M12):

  | Outcome | When (first rule wins) |
  |---|---|
  | `already_undone` | every claim of the operation on the item is already withdrawn |
  | `kept_holdout` | a live `holdout_frozen` newer than the operation's class claim: the withdraw is ignored for class while frozen (M1) |
  | `kept_changed` | a newer live claim of the same layer and the same or higher actor rank decides the item |
  | `kept_stale_class` | the fold without the operation's claims gives `class_unresolved` (renamed away or deleted) |
  | `restored` | otherwise |

  A step-back undo on one item withdraws every live claim of the person's newest live operation on
  it, so a discard (`class_cleared` + `review_dismissed`) or a region edit comes back whole; the walk
  of `find_undo_snapshot` (`src/domain/history.py:219`) becomes one indexed read. As today, the walk
  steps back over label, batch label, move, resolve and discard operations and not over a bulk accept
  (`HUMAN_LABEL_WRITERS`; Q4). A redo appends
  copies under a new operation; a `withdraw` is never withdrawn; two undos of one claim meet on
  `uq_claims_withdraws` (409 `already_undone`).
- **Operations**: every human act writes one `label_operations` row (kinds in 2.2) and its claims
  carry `op_id`; a resumed chunk writes nothing twice (M4). Single labels too: rows of kind `label`
  beyond the newest 1,000 per project are pruned (m4; today's cap, `label_ops/store.py:55-58`); their
  claims stay, which is why `claims.op_id` has no foreign key.
- **Import**: a validated label is a `class_set` with `actor_kind = 'import'`, `actor =
  'import:<id>'` and the import operation's `op_id` (naming `dataset_import_id`). It ranks below
  every live human claim (B2); where a person decides, the import reports `locked_conflict` as today
  (`dataset_import/item_labels.py:30-40`), from `fold.conflicts`. On a held-out item an import or
  combine class claim written after the freeze does not decide, unless it corrects the class the
  import itself set (today it is refused, `item_labels.human_owned`; Q3); the writer still reports
  the conflict. A `label_trust: suggestion` label is
  a `class_opinions` row (`import_suggestion`, `import_id`); its boxes and detections carry
  `import_id` under an `import` analyser. Import undo withdraws the operation's claims and deletes the
  rows with its `import_id`: items only the import created disappear, items a person claimed stay
  (`is_human_owned_item`, `occ_locks.py:154`). Combine is the same with `actor_kind = 'combine'`.
- **Retention**: claims none (R3; erasure, section 8). A labelling hour labels about 2,000 items and
  writes about 2,100 claims and 5-20 operations; a 10 % holdout freeze writes 1 M claims at 10 M.
  Twenty million claims over a 10 M project's life is the sizing figure.

## 4. The fold

### 4.1 Inputs and outputs

`fold_item(inputs) -> ItemState | None` in `src/domain/fold.py`, pure (no I/O, no clock), with
`fold_boxes`, `fold_photo`, `fold_entity` and `conflicts` beside it. Inputs for one item: its `items`
row (with `geometry_seq`, `region_seq`); its detections; its class opinions; its region result and
machine boxes with readings; the photo's `frame`; the live claims attached to it (after
`claim_match`), with the withdrawn ones the holdout rule still needs (layer 2); the class registry
snapshot. Output: the `item_state` row and the item's `box_state` rows, or `None` when the item has
neither a current detection nor a live asserting claim (its `items` row is then deleted in the same
transaction). Cluster placement is not an input: readers join it (layer 5).

The step before the fold, `attach(photo)`, runs `claim_match` and inserts or deletes `items` rows
under the photo lock (section 7); it is the only writer of `items` rows' existence.
`fold.rebuild(project)` (M11) holds a job lease (`src/clients/job_lease.py`) and refolds in chunks
of `item_key` order, each chunk under the same item lock order as every writer: it upserts
`item_state` and `box_state` rows (writing only rows that differ) and deletes state rows whose item
has no identity left. No table is emptied while readers run; the revision counters come from claims
and the stored sequences, so a rebuild reproduces them.

### 4.2 Machine class precedence (today's behaviour, then the function)

Today the effective machine class is whatever automated writer wrote last, limited by guards and by
the queries that choose which items each writer sees:

- No automated class write touches a locked or validated item (`class_write_locked`,
  `src/services/curation/class_write_guard.py:66`; region worker `scripts/curation/worker/crop_gate.py:54`,
  propagation `assist_loop/propagate.py:87`).
- The VLM is not asked when a classifier is at or above the skip confidence, or the item already
  holds `vlm`, `vlm_unmatched`, `vlm_new_class_pending` or `assist_propagated`
  (`scripts/curation/vlm_pending.py:66-113`; the worker's combined call,
  `scripts/curation/worker/combined_resolve.py:76-83`, threshold 0.80; auto-label,
  `src/services/curation/autolabel/selection.py:84-100`). When asked, its answer replaces whatever
  machine class was there.
- An `open_vocab_target` class is kept against the VLM, whose answer becomes the name-only suggestion
  (`keep_target_class`, `src/services/curation/vlm_class_attempt.py:200-217`).
- Propagation overwrites eligible members' machine classes, but "a dissenting one keeps its own
  answer" (`assist_loop/propagate.py:1-10`).
- An import suggestion replaces a machine class at import (`dataset_import/item_labels.py:38-40`
  refuses only human and holdout items); the VLM still selects it later and replaces it.
- `cluster_majority_agreement` and `classifier_vlm_agreement` have readers but no writer (`rg`:
  `vlm_pending.py:105-106`, `combined_resolve.py:81`, `stats_dataset.py:84`): no rank; the lane
  removes the names.

`machine_class(opinions, detection)` picks the first present of, in order:

| Rank | Source | Wire `class_source` | Reason (today) |
|---|---|---|---|
| 1 | `open_vocab_target` | `open_vocab_target` | the user's own word for what the prompt finds; kept against the VLM |
| 2 | `vlm` (outcome `matched`, `reclassified`, `unmatched`, `new_class_pending`) | `vlm`, `vlm_reclassified`, `vlm_unmatched`, `vlm_new_class_pending` | the last writer over every source below it; never re-asked once written; `unmatched` clears the class as `unmatched_class_clear` does (`class_sources.py:233`) |
| 3 | `import_suggestion` | `external_label`, validated false | written over machine classes at import; replaced by the VLM later |
| 4 | `assist_propagated` | `assist_propagated` | written over every eligible machine class, including a confident classifier (`member_state`, `assist_loop/partition.py:166-177`); an item's own VLM answer is kept |
| 5 | `classifier` (newest row) | `<profile>_model`, `<primary>_low_conf` | written at ingest; its confidence only decides whether the VLM is asked (scheduling, not precedence) |
| 6 | `adopted_detector` | `detector` | `POST /label_ops/adopt_detector` writes only on an item that had no class (`class_label.py:110-122`) |
| 7 | the detection itself | `<primary>_proposal`, `open_vocab_proposal`, `unlabeled_proposal` | |

Inside one source there is one row by key. One order is undefined today (whichever of propagation
and an import suggestion wrote last): rank 3 over 4 here, because an import label is about this item
and a propagated one is carried over from its cluster (plan 15 section 14, answer 5). The function
skips every opinion and detection whose (source, analyser) pair a live deciding `class_cleared`
lists in `hidden` (layer 2, Q10), and an opinion row without a class never displaces one with a class
(section 2.3, M17).

### 4.3 The fold, layer by layer

**Ranking, in every layer (B2).** A layer's live claims are ordered by actor rank (`human` above
`import` and `combine`), then `claim_id`; the first is the **deciding** claim. `fold.conflicts` lists
the items where an import or combine claim lost to a human one (`locked_conflict`,
`combine_conflict`).

1. **Geometry (R3 pin, B1, Q11)**: a deciding `item_drawn`/`geometry_set` gives the bbox and
   `geometry_source` (`human`; `import` for an import or combine actor). Else, exactly while today's
   `is_locked_item` holds (3.2: a person's or validated-import class, holdout, human or import
   geometry, a human delete, any human-owned box, a set validated by a person or import), the box the
   person saw: the item's box as shown at the newest live claim that pins (its `subject_bbox`, or the
   `item_bbox` of a box or region claim), whoever made it; `geometry_source` unchanged and `geometry_pinned` true. A clear, a
   dismissal, a flag, a move to a cluster and an exclusion for another reason do not pin. Else the machine geometry (3.4 step 6). Later
   detections only add rows. `area`, `rank` (by area, `item_geometry.py:219`) follow.
2. **Class**: the deciding claim among `class_set`, `class_cleared`, `class_moved`:
   - `class_set` → its class (3.3), `class_source` `human` (`external_label` for import or combine).
     An unresolvable name: `class_id` NULL, `class_unresolved`, same `class_source`, still locked (M17).
   - `class_cleared` → the machine class of 4.2 over the opinions and detections its `hidden` list
     does not name, not validated: earlier class claims are dead, an opinion from an analyser the
     person did not see shows (M17, Q10; `label_undo.py:109-121`).
   - `class_moved` → the machine class as a suggestion, not validated (`class_label.py:239-277`).
   - **Holdout (M1)**: while a live `holdout_frozen` exists, a withdraw written after the freeze of a
     class claim written before it is ignored; with no deciding claim the class is the payload's
     `machine_class`, none when `null`, and no opinion applies. A person's class claim newer than
     the freeze still decides (no route refuses it today, `crops_write.py:39`; section 11, question 1).
     An import or combine class claim written after the freeze decides only when it corrects the
     class the import itself set (Q3). The freeze payload records what a clear had left, never what
     it hid.
   - Otherwise `machine_class` of 4.2.
   - `class_validated` = a resolved deciding `class_set` **and not excluded** (M2);
     `class_claim_id`, `class_op_id` (M8), `class_opinion_source`, `class_confidence` name the source.
3. **Flags**: `excluded*` from the deciding `excluded`/`restored` (unexclude writes `restored`, undo
   withdraws; the same state, because validation is computed: M2); `review_dismissed_at` from
   `review_dismissed`/`review_restored`; `holdout` = a live `holdout_frozen`; `needs_new_class` = a
   live flag or opinion outcome `new_class_pending` until a later human `class_set` (a plain label too, Q1);
   `vlm_suggestion_dismissed` = a live `vlm_dismissed` naming the current VLM suggestion
   (`class_sources.py:198`); `probe_dismissed_class` from `probe_dismissed`; `audit_*` from
   `audit_sampled` and the latest human class verdict after it (`src/domain/audit_math.py`; today's
   rule, Q2); `combine_conflict` =
   a live conflict claim with no later `class_set`; `import_ids` from live claims, not-rebuildable
   opinions and detections (M13).
4. **Boxes (B3)**: per box the deciding claim of each aspect (geometry, state, text) wins over the
   machine box; `box_deleted` removes and never creates; claim-created boxes have `source`
   `human`/`import`; a human reject sets `REJECT_REASON_HUMAN`, a typed text `text_source = 'human'`;
   else the reader consensus of `box_texts` (`src/services/detection/region_text.py:400-455`, moved
   to `src/domain/`). **Closed set**: with C the newest live human or import box or region claim, the
   set is closed when no box is `proposed` after folding the claims with the machine boxes of the
   generation the person saw (the newest generation not newer than C's `region_seq`; the repository
   keeps that generation when a re-run replaces the rows, Q9); then later unclaimed machine boxes
   stay results, unfolded (today the locked item is never written, `occ_locks.py:137-151`). A live
   `region_status_set` newer than every box claim folds no machine box. `region_status`: the
   verdict's own status (`no_region_visible`, or the `verify_rejected` / `false_positive` the person
   gave over an empty list, Q7); else, with a
   human or import box claim, `derive_status(boxes, empty_status=NO_REGION_VISIBLE)`
   (`region_writes.py:68-105`); else the machine status. `region_validated` = closed (today's rule,
   `region_writes.py:100-102`); `region_set_complete` = closed, else the machine region's own
   (Q8, an `item_state` column); counts and max score as `boxes_write_fields`
   (`region_boxes.py:329-382`); `region_verifier` from C's actor, else the machine's.
5. **Placement (M6, M21)**: `moved_partition_id`, `moved_cluster_uid` from the deciding `class_moved`.
   Readers use them only while that partition serves, else the serving partition's membership, so an
   old move never shows a cluster the serving partition lacks; its class effect stays.
6. **Revisions (M11)**: `geometry_revision` = the item's pinning claims and their withdraws plus
   `items.geometry_seq`; `region_revision` = its box and region claims and their withdraws plus
   `items.region_seq`. Both only grow and a rebuild reproduces them (claims and sequences are record
   data); `expected_region_revision` and the thumbnail `?v=` (`wire.py:470`) rely on that.
   `updated_at` = the newest input `at`/`created_at`, stored only when another column changed (M5).

### 4.4 Each lock predicate is implied by the data shape

Machine writers insert or replace only result rows (section 2.3); they never write `claims`, and
the fold puts every claim layer above every result layer. So for each predicate of
`src/clients/occ_locks.py`, the state it protects comes from a claim, which no machine write can
change:

| Predicate (`occ_locks.py`) | Today true when | In the record it is true when | Why no machine write can change the protected state |
|---|---|---|---|
| `is_human_marker` (:22) | a guard field contains `human` | `class_source` is `human` because layer 2 found a deciding human `class_set` | layer 2 reads claims before opinions; opinions cannot remove a claim |
| `is_locked_class` (:44), human | `class_source` human | as above, resolved or `class_unresolved` | as above |
| `is_locked_class`, import | `external_label` and validated | deciding `class_set` with `actor_kind = 'import'` | as above; it ranks below any human claim (B2); an import suggestion is an opinion and stays replaceable, as today |
| `is_locked_class`, holdout | `test_holdout` | live `holdout_frozen` | layer 2 uses the frozen class (or none) before any opinion and ignores later withdraws of older class claims |
| `is_locked_box` (:77) | box `source` human, human reject, human text, or import box not proposed | the box aspect comes from a live box claim | layer 4 reads box claims before `region_boxes` and `box_texts` |
| `is_human_deleted` (:90) | excluded, by human, `not_an_object` | live human `excluded` with that reason | layer 3; a dormant claim re-applies to a new matching detection (3.4) |
| `has_human_geometry` (:102) | `geometry_source` human or deleted | live human `item_drawn`/`geometry_set`, or deleted | layer 1 reads claims before detections |
| `has_import_geometry` (:109) | `import_ids` and validated `external_label` | live import `item_drawn` or `class_set` | layer 1 (pin) and 2 |
| `is_locked_geometry` (:121) | the two above or holdout | the claims above or live `holdout_frozen` | layer 1 pins the box to the claim's `subject_bbox` or the `item_bbox` it saw (Q11); `claim_match` step 4 keeps the item |
| `is_locked_item` (:137) | any of the above, any human-owned box, or region validated by human/import verifier | any of the above, a live human box claim, or a closed box set; `geometry_pinned` is exactly this predicate (Q11) | layer 4: later unclaimed machine boxes stay out of a closed set; layer 1 keeps the item's box |
| `is_human_owned_item` (:154) | human class, human geometry, curator holdout, human box, human-validated set | the item has a live claim with `actor_kind = 'human'` | used by import undo: the item survives the withdrawal |

The wire flags `label_locked` and per-box `locked` (`wire.py:389`, `:525`) keep today's predicates,
evaluated on the folded state; one difference is intended: a human accept of a machine box is now a
claim, so no machine writer can touch it, while the wire `locked` stays false for it as today
(`region_boxes.py:283-305`).

### 4.5 Tests the lane writes

Hypothesis property tests on the pure fold (`tests/domain/test_fold_properties.py`) and database
tests marked (DB) (`tests/db/`), each named with the finding it closes (appendix A).

| Test | Property |
|---|---|
| `test_machine_write_order_irrelevant`, `test_lock_predicates_hold` | any permutation of machine writes around the same claims gives the same state; each predicate of 4.4 is unchanged by any further machine write |
| `test_withdraw_leaves_no_trace` | fold(claims + c + withdraw(c)) = fold(claims), outside a freeze, revisions aside |
| `test_pin_survives_machine_writes` (B1) | box and class of an item with a live human, import-validated or holdout claim are invariant under any machine writes, including a re-detect overlapping at IoU ≥ 0.5 |
| `test_human_outranks_import_and_combine` (B2) | import or combine class claims in any order around a human claim leave the human class; `conflicts` reports each |
| `test_whole_set_status_expands` (B3) | per whole-set status, the folded boxes equal `boxes_with_status` over the boxes shown; `region_validated` follows today's rule; a later unclaimed machine box stays out of a closed set; `box_deleted` never creates |
| `test_undo_chunks_atomic` (B4, DB) | an undo killed after any chunk leaves every `item_state` row equal to a fresh fold; the resume reaches `undone` with no duplicate withdraw |
| `test_erasure_catalogue_complete`, `test_erasure_leaves_nothing` (B5, DB) | section 8 |
| `test_holdout_freezes_class` (M1) | an item frozen as "no class" gets none from later opinions; a later withdraw of an older class claim changes nothing and counts `kept_holdout` |
| `test_excluded_never_validated` (M2) | no history yields `excluded` and `class_validated` together |
| `test_resumed_chunk_no_duplicate` (M4, DB) | a re-run committed chunk inserts nothing; a later step-back undo clears the item |
| `test_undo_outcomes_match_effect` (M12) | dry-pass counts equal the observed change per item; a step-back undo of a discard restores class and review state together |
| `test_cleared_hides_only_the_analysers_it_saw`, `test_empty_vlm_answer_keeps_class`, `test_unresolved_keeps_lock` (M17, Q10) | the three M17 rules; a clear hides the (source, analyser) pairs it lists, a newer analyser's opinion shows |
| `test_rerun_of_the_same_analysers_after_a_clear_leaves_it_cleared` (Q10) | emptying every result and re-running the same analysers restores a cleared item |
| `test_what_locks_the_item_pins_its_box`, `test_clears_dismissals_flags_moves_and_other_exclusions_do_not_pin` (Q11), truth table `p_pinned` | the box is pinned iff today's `is_locked_item` holds, on every enumerated history |
| `test_rerun_restores_state`, `test_rebuild_equals_incremental` (M11, DB) | emptying rebuildable results and re-running (detections jittered under IoU 0.5) restores every claimed item, revisions aside; a chunked rebuild during writes ends equal to incremental folding, revisions included |
| `test_class_resolution` (M20) | follows `merged_into`, survives a rename; a foreign hint never selects a class; an unknown name gives `class_unresolved` |
| `test_region_revision_increases`, `test_opinion_rank` | `region_revision` grows on every observable write; removing the top opinion yields the next rank |
| `test_frame_mismatch_dormant` (M14), `test_old_move_not_placed` (M21) | a claim group of another `frame` attaches to nothing and is reported; an item moved in an old partition is placed by membership |

## 5. Field map

Classes: M machine, H human, I identity or provenance, D derived. "read" = derived at read from the
named source. Today's fields from inventory 1.1 and 1.2, checked against
`src/clients/curation_opensearch/bodies_core.py`, `items_extra.py`, `capture_fields.py`,
`assist_fields.py`, `open_vocab_fields.py`, `src/clients/opensearch/indexes.py`.

### 5.1 `images` document (62 fields)

| Today's field | Class | Destination |
|---|---|---|
| image_id, imohash, width, height, decode_route, source, source_identifier | I/M | `photos.` same names (`imohash` as bytes) |
| image_path | I | `photos.storage_key` (R9); the wire shows its display form (section 10) |
| ingest_run_id | I | `photos.ingest_run_tag` |
| indexed_at | I | `photos.created_at` |
| original_resolution | D | read: `width x height` (written as `f'{width}x{height}'`, `item_doc.py:153`) |
| pe_embedding | M | vector table of the frame space (section 6) |
| embedding | ? | deleted: no writer (`rg` finds only the mapping, `bodies_core.py:120`) |
| non_jpeg_format_skipped | ? | deleted: no writer (`rg`: only `bodies_core.py:118`) |
| error_kind | I | deleted: a photo is stored whole or not at all; a failure is the job's or `ingest_source_files.error` |
| item_ids, ingest_write_id, ingest_incomplete, repair_started_at | I/D | deleted: repair lease and item list go (plan 15 section 5) |
| capture.taken_local, taken_utc, offset_minutes, time_source, offset_source, time_precision, camera_make, camera_model, lens_model, focal_length_mm, focal_length_35mm, orientation, has_location, exif_status, altitude_m, location_accuracy_m, heading_deg, location_precision (18) | M | `photo_capture.` same names |
| capture.location | M | `photo_capture.lat`, `lon` |
| capture.parser | I | `photo_capture.analyser_id` (analyser `capture_parser`, `capture@1`) |
| dataset_split, import_source_stem, import_stratum, import_hard_negative, import_label_state, negative_for | I | `photo_provenance.dataset_split, source_stem, stratum, hard_negative, label_state, negative_for` |
| import_ids | I | read: `photo_provenance.import_id` rows |
| origin_project, origin_image_id, origin_split | I | `photo_provenance.origin_project_slug, origin_image_id, dataset_split` (kind `combine`) |
| near_dup_group_id, near_dup_leak_group_id, near_dup_run_id, near_dup_is_representative, near_dup_group_size | D | read: `near_dup_members` / `near_dup_groups` of the current run (02d section 3) |
| open_vocab_status, open_vocab_status_at | I | `analysis_ledger.state`, `updated_at` (kind `open_vocab`) |
| counts.items, labeled, validated, unlabeled, in_holdout | D | read: aggregate of `item_state` by `photo_key` (summary candidate, section 2.6) |
| counts_at | D | deleted: bookkeeping of the Valkey dirty set, which goes |

### 5.2 `items` document (top level)

| Today's field | Class | Destination |
|---|---|---|
| crop_id; image_id; created_at; region_box_seq | I/D | `items.crop_id` (copy in `item_state`); `items.photo_key` → `photos.image_id`; `items.created_at`; `items.box_seq` |
| request_id | I | `photos.ingest_request_id`; a drawn item's in `claims.payload.request_id` |
| image_path, source | I | read: `photos.storage_key`, `photos.source` |
| bbox_norm, crop_area_norm, crop_rank_in_image, geometry_source, geometry_revision | M/H/D | `item_state.bbox, area, rank, geometry_source, geometry_revision` (fold layer 1) |
| detector_bbox_norm | M | `detections.bbox`; at claim time `claims.subject_machine_bbox` |
| class_id, class_source, label_source, class_validated, class_labeled_at, confidence | H/M | `item_state.` same names, `confidence` as `class_confidence` (fold layer 2) |
| updated_at, test_holdout, needs_new_class, review_dismissed_at, import_ids, combine_conflict, audit_sample, audit_outcome | D/H | `item_state.updated_at, holdout, needs_new_class, review_dismissed_at, import_ids, combine_conflict, audit_sample, audit_outcome` (`import_ids` from claims, opinions and detections, M13) |
| class_excluded, excluded_at, excluded_reason | H | `item_state.excluded, excluded_at, excluded_reason` |
| region_status, region_validated, region_label_source, region_verifier, region_count, region_rejected_count, region_max_score, region_revision | D/H | `item_state.` same names (`region_verifier`: human/import from a claim, else the machine's) |
| cluster_partition_id, cluster_uid, cluster_state, cluster_score, cluster_margin, cluster_assigned_by | D | read: `cluster_memberships` of the serving partition; `item_state.moved_*` with `assigned_by = 'edit'` while the move's partition serves (layer 5) |
| capture_taken_local, capture_taken_utc | D | read: the photo's effective capture time (`photo_state`, else `photo_capture`) |
| probe_disagreement; probe_pred_entropy | D/M | read: `probe_predictions.class_name` against the effective class, null outside the probe's class names; `probe_predictions.entropy` |
| class_name | H/M | read: `project_classes.name` by `item_state.class_id` |
| class_detector, class_detector_version, class_labeler | I | read: the deciding claim's payload `labeler`, `profile` (M19) or the deciding opinion's analyser |
| label_op_id | I | `item_state.class_op_id` (the deciding claim's or opinion's operation, M8) |
| detector_class_name, detector_class_id, detector_confidence, proposal_name, blur_lap_var, blur_lap_ratio, blur_full_var, source_prompt, mask_polygon | M | `detections.class_name, class_id, confidence, proposal_name, blur_lap_var, blur_lap_ratio, blur_full_var, prompt, mask_polygon` (joined, no copies) |
| proposed_by_import | I | `detections.import_id` |
| open_vocab_set, open_vocab_revision | I | read: the open-vocabulary detection's `analysers` row |
| vlm_confidence, vlm_raw_label, vlm_raw_label_conf, vlm_raw_class, vlm_proposed_class, vlm_class_attempted_at, vlm_class_empty_reason, vlm_verify_completed_at | M/I | `class_opinions.confidence_cat, raw_label, raw_label_conf, raw_class, proposed_class`, `analysis_ledger.updated_at` (kind `vlm`, M17), `class_opinions.empty_reason`, `created_at` where `via = 'combined'` |
| vlm_endpoint, vlm_model, vlm_prompt_pack | I | read: the VLM opinion's `analysers` row (name, version, config) |
| vlm_label_cluster_id, vlm_label_cluster_name, vlm_label_cluster_distance | M | `raw_label_clusters.cluster_id, cluster_name, distance`, joined on `raw_label` |
| vlm_dismissed_class_id, vlm_dismissed_class_name, vlm_dismissed_at | H | `claims` (`vlm_dismissed` payload, `at`); flag `item_state.vlm_suggestion_dismissed` |
| needs_new_class_note, needs_new_class_at (written but unmapped, `crops_write.py:237-239`) | H | `claims.payload.note`, `claims.at` |
| probe_dismissed_at, probe_dismissed_class, probe_dismissed_model_version | H | `claims` (`probe_dismissed`); `item_state.probe_dismissed_class` |
| audit_batch_id, audit_label_name, audit_label_source, audit_sampled_at | I | `claims` (`audit_sampled` payload, `at`) |
| audit_human_name | H | read: class of the first `class_set` after the sample |
| excluded_by, review_dismissed_by | I | read: the claim's actor |
| near_dup_group_id, near_dup_leak_group_id, near_dup_run_id, near_dup_is_representative, near_dup_group_size | D | read: `near_dup_members` / `near_dup_groups` of the current run |
| pe_embedding; vec_{key}; region_box_embeddings.box_id, .bbox_norm, .embedding | M | rows of the item space's, that space's and the box space's vector tables (section 6); `bbox_norm` becomes `input_rev` |
| embedding_state, vec_{key}_state | D | read: a row in the vector table, else `vector_gaps.reason` |
| region_reason, region_verified, region_verified_at, region_verifier_version, region_auto_confirmed, region_visible, region_skip_verify, region_gate_skip, region_detector_chain, region_detected_at, region_set_complete, region_profile, region_profile_revision | M/I | `item_region_results.reason, verified, verified_at, verifier_version, auto_confirmed, visible, skip_verify, gate_skip, detector_chain, detected_at, set_complete`; profile and revision in its `analyser_id` |
| region_rejection_reason | D | read: the highest-scoring rejected `box_state` row when no live box (`region_boxes.py:369-379`) |
| region_boxes | M/H | `region_boxes` + `box_state` (5.3) |
| item_text_lines | M | `text_lines` rows with `item_key` |
| probe_pred_class, probe_pred_class_id, probe_pred_confidence, probe_pred_margin, probe_scored_at, probe_model_version | M/I | `probe_predictions.class_name, class_id_hint, confidence, margin, scored_at, analyser_id` |
| uniqueness_score, uniqueness_scored_at, mistakenness_score, mistakenness_scored_at, uniqueness_method, uniqueness_version, mistakenness_method, mistakenness_version | M/I | `item_scores.score, scored_at` by kind; method and version in `analyser_id` |
| viz_x, viz_y, viz_projection_version | M/I | `item_projections.x, y, analyser_id` |
| assist.run_id, run_created_at, class_name, class_confidence, attributes, boxes, vlm_endpoint, vlm_model, prompt_pack, prompt_pack_revision | M/I | `assist_answers.run_id, run_created_at, class_name, class_confidence, attributes, box_attributes`; the last four in `analyser_id` |
| dataset_split, import_source_stem, import_stratum, import_hard_negative, imported_at, import_dataset_name, import_dataset_sha, on_negative_frame | I/D | read: `photo_provenance` (split, stem, stratum, hard negative, `label_state = 'negative'`); earliest of the item's import claims, opinions and detections (M13); `dataset_imports.dataset_name`; `dataset_uploads.sha256` |
| proposal_chain | I | `item_notes` rows of kind `proposal` (M19); the disagreement records `class_mismatch`, `missed_label`, `unmatched_detection` are `item_notes` too |
| import_standalone_region | I | `claims.payload.standalone_region` |
| origin_project, origin_item_id, origin_image_id; origin_split | I | `claims.payload.origin` of the combine-actor claims; split read from `photo_provenance` |
| combine_conflict_origins, combine_merged_origins | I | `claims.payload` of `combine_conflict` |
| class_id_history, edit_history | H/I | deleted: claims are the history |
| label_validated | ? | deleted: no writer (`rg`: only readers and filter names); wire value read as `class_validated OR region_validated` (`wire.py:337`) |
| region_status_before_requeue | I | deleted: a requeue is a re-run; the previous result is replaced, not kept |
| region_class_id | ? | deleted: written only by the 0.6.0 upgrade (`upgrade/from_060/steps/legacy_repair.py:107`) |
| region_pairing | ? | deleted: no writer (`rg`: only `region_fields.py:99`, the wire model, preflight labels) |
| region_run_id, region_run_verify | I | deleted from item storage: the assist segment run's item list and verify choice become its `job_items` and job params (02a section 5) |
| item_text_tokens | D | deleted: the trigram index on `text_lines.text` answers the same search |
| excluded_prior_class_validated | H | deleted: the fold computes `class_validated` from the class claim and the exclusion (M2) |

### 5.3 `region_boxes[]` element (32 keys of `RegionBox`, `src/domain/region_boxes.py:41-80`; the inventory counted 27)

| Key | Class | Destination |
|---|---|---|
| box_id | I | `region_boxes.box_id` / claim `subject_box_id`; `box_state.box_id` |
| bbox_norm, state, rejection_reason, bbox_correct | M/H | `box_state.` (fold of `region_boxes.bbox, state, rejection_reason` and box claims) |
| score, detector, detector_version, confidence, detected_at | M | `region_boxes.` same names (`box_state` copies) |
| source | I | `box_state.source` (`human`/`import` from a claim, else `region_boxes.source`) |
| text, text_source, text_choice, text_disagreement | H/M/D | `box_state.` same names |
| text_raw, text_frame, text_confidence, text_conf_char_min, text_reader, text_reader_version | M | `box_texts.text_raw, text_frame, confidence, conf_char_min, analyser_id (×2)` |
| text_vlm, text_ocr | M | `box_texts.text` of reader `vlm`, `ocr` |
| text_vlm_invalid | M | `box_texts.invalid` |
| text_norm | D | deleted: `text_fold` serves exact and fragment match (`bodies_core.py:220-226`) |
| text_fold | D | `box_state.text_fold` |
| cluster_partition_id, cluster_uid, cluster_state, cluster_score, cluster_margin, cluster_assigned_by | D | read: `cluster_memberships` of the serving box partition |

### 5.4 Global indexes (1.2)

| Index: field | Class | Destination |
|---|---|---|
| global: image_id, imohash, width, height | I/M | `photos.` (the core project, plan 15 section 14, answer 1) |
| global: image_path | I | `photos.storage_key` |
| global: file_size_bytes | I | `photos.size_bytes` |
| global, vehicles, people, faces, ocr: metadata (5) | I | `photos.client_metadata` |
| global, vehicles, people, faces, ocr: indexed_at (5) | I | `photos.created_at` / `items.created_at` / face and line rows' photo |
| global, vehicles, people, faces: clustered_at (4); faces: cluster_assigned_at | I | read: `cluster_partitions.finished_at` of the serving partition |
| global: global_embedding; vehicles, people, faces: embedding (3) | M | vector tables: frame space, item space, face space of the core project |
| global, vehicles, people, faces: cluster_id, cluster_distance (8) | D | read: `cluster_memberships.cluster_uid`, `score` (the integer id goes, X-1 of 02d) |
| global: near_dup_group_id, near_dup_run_id, near_dup_is_representative, near_dup_group_size, near_dup_cosine_to_rep | D | read: `near_dup_members` (`cosine_to_rep`) and `near_dup_groups` |
| vehicles, people: detection_id, image_id, image_path (6) | I | `items.crop_id`, `items.photo_key`, read `photos.storage_key` |
| vehicles, people: box (2) | M | `detections.bbox` |
| vehicles: class_id, class_name, confidence | M | `detections.class_id, class_name, confidence` |
| people: has_face, face_id | D | read: `faces.person_item_key` |
| faces: face_id, image_id, image_path | I | `faces.face_id`, `faces.photo_key`, read `photos.storage_key` |
| faces: person_detection_id | I | `faces.person_item_key` |
| faces: box, landmarks, confidence, quality_score | M | `faces.bbox, landmarks, det_score, quality` |
| faces: person_id | D | `face_state.person_entity_key` |
| faces: person_name, named_at, merged_at, merged_from | H | `claims` (`entity_named`, `entity_merged`); `entity_state.name, merged_into` |
| faces: is_reference | ? | deleted: no writer (`rg`: mapping `indexes.py:307` and the reader `persons.py:311` only) |
| faces: thumbnail_b64 | M | deleted: a derivative, served from the crop and thumbnail cache (plan 15 section 8) |
| ocr: ocr_id | I | `text_lines.line_key` |
| ocr: image_id, image_path | I | `text_lines.photo_key`, read `photos.storage_key` |
| ocr: text, text_raw, box, box_normalized, det_score, rec_score, language | M | `text_lines.text, text_raw, polygon, box_norm, det_score, rec_score, language` |

### 5.5 Counts by destination

"To a column" includes a row of a result, claim or vector table (a vector becomes a row of its
space's table; a human value becomes a claim's payload or time).

| Group | Fields | To a column or row | Derived at read | Deleted |
|---|---|---|---|---|
| `images` (5.1) | 62 | 42 | 12 | 8 (`embedding`, `non_jpeg_format_skipped`, `error_kind`, `item_ids`, `ingest_write_id`, `ingest_incomplete`, `repair_started_at`, `counts_at`) |
| `items` top level (5.2; the 2 unmapped `needs_new_class_*`, 10 `assist.*` leaves, 3 `region_box_embeddings` leaves and the `vec_{key}` pair counted one each) | 175 | 124 | 41 | 10 (`class_id_history`, `label_validated`, `region_status_before_requeue`, `region_class_id`, `region_pairing`, `region_run_id`, `region_run_verify`, `item_text_tokens`, `excluded_prior_class_validated`, `edit_history`) |
| `region_boxes[]` (5.3) | 32 | 25 | 6 | 1 (`text_norm`) |
| global indexes (5.4: 17 + 13 + 12 + 22 + 12) | 76 | 50 | 24 | 2 (`is_reference`, `thumbnail_b64`) |
| **total** | **345** | **241** | **83** | **21** |

Every "no writer found" field of inventory section 9 was re-checked with `rg` on this tree:
`embedding` and `non_jpeg_format_skipped` (images), `label_validated` (items), and additionally
`region_class_id`, `region_pairing`, `is_reference`: deleted. `region_visible` has a writer
(`scripts/curation/worker/verify.py:207`) and is kept. The `OP_REGION_FIELD_*` storage-name override
of `RegionFields.from_env` (`src/config/region_fields.py:117`) has no meaning for fixed columns; the
lane deletes it and lists the variables in `src/config/retired_env.py`.

## 6. Vectors

### 6.1 Template (one table per embedding space), and the role that creates it

```sql
-- schema vec, created by the migration (owner op_owner) with CREATE granted to op_vector only;
-- name 'vec.v<projects.lock_key>_<space_key>' (≤ 60 chars), created by the embedding-space job as op_vector
CREATE TABLE vec.v<lock_key>_<space_key> (
  subject_key bigint PRIMARY KEY, PROJECT,   -- item_key, box_key, photo_key or face_key by embedding_spaces.target
  input_rev   integer NOT NULL DEFAULT 0,   -- the item's geometry_revision (item space) or region_revision (box space) it was computed from
  embedding   halfvec(<dim>) NOT NULL
);
ALTER TABLE vec.v<lock_key>_<space_key> ALTER COLUMN embedding SET STORAGE PLAIN;
ALTER TABLE vec.v<lock_key>_<space_key> ENABLE ROW LEVEL SECURITY, FORCE ROW LEVEL SECURITY;  -- + policy RLS(02a)
GRANT SELECT, INSERT, DELETE ON vec.v<lock_key>_<space_key> TO op_app;
-- fillfactor 100: a row is inserted, replaced whole (DELETE + INSERT on a new build), never updated in place
-- The approximate index is gate B's: e.g. CREATE INDEX CONCURRENTLY ... USING hnsw (embedding halfvec_cosine_ops)
```

**The role (plan 15 section 14, answer 2; M10).** `op_vector`: a login role, `NOBYPASSRLS`, granted
to nobody, with its own secret and a small autocommit pool used only by
`src/services/embedding_spaces/ddl.py`. Its timeouts are set on the role (`statement_timeout = 0`,
`lock_timeout = 0`), which applies at login, whereas `SET ROLE` "does not process session variables
as specified by the role's ALTER ROLE settings" ([SET ROLE](https://www.postgresql.org/docs/18/sql-set-role.html));
autocommit, because `CREATE INDEX CONCURRENTLY` "cannot be executed inside a transaction block"
([CREATE INDEX](https://www.postgresql.org/docs/18/sql-createindex.html)). It owns its tables;
`FORCE` binds an owner without `BYPASSRLS` ([row security](https://www.postgresql.org/docs/18/ddl-rowsecurity.html)),
but an owner could turn `FORCE` off, so the job asserts `relrowsecurity` and `relforcerowsecurity`
before `ready` and the schema test asserts them on every `vec` table. Tests: `op_app` cannot create,
alter or drop in `vec`; only `ddl.py` reads the secret; a build killed midway is recovered (6.3).

No foreign key to the subject: a subject deleted by the attach step (section 4.1) deletes its
vectors in the same transaction through the repository (one `DELETE ... WHERE subject_key = ANY`),
because the template cannot know which table its key points to; erasure reaches the tables through
its catalogue (section 8). A check job (`fold.verify_orphans`) counts vectors without a subject. The
model version is the space's (`embedding_spaces.model_version_id`): a space is one model version by
definition (`src/domain/embedding_spaces.py:1-6`), so R6's "version beside it" is the space row.

### 6.2 Storage strategy, decided from the documentation

pgvector declares `halfvec` with `STORAGE = external` ([sql/vector.sql](https://github.com/pgvector/pgvector/blob/master/sql/vector.sql)).
A row is toasted when it is "wider than TOAST_TUPLE_THRESHOLD bytes (normally 2 kB)"
([TOAST](https://www.postgresql.org/docs/18/storage-toast.html)). Our row: 24 header + 8 + 16 + 4 +
`halfvec(1024)` of 2 x 1024 + 8 = 2,056 bytes ([pgvector](https://github.com/pgvector/pgvector):
"2 * dimensions + 8 bytes") = 2,108 bytes, just above the threshold. The choices:

| Strategy | What the docs say | Cost here |
|---|---|---|
| EXTERNAL (default) | "allows out-of-line storage but not compression" | every vector moves to the TOAST table in two chunks (about 2 kB and 60 bytes); main heap 10 M x 74 B ≈ 0.7 GB, TOAST about 2.6 M pages ≈ 21.5 GB plus its chunk index ≈ 0.5 GB: **22.7 GB**. An exact scan or a `COPY TO` reads each vector through the chunk index (two extra page reads per row). pgvector warns the planner does not count TOAST in its estimates and may choose a worse serial scan |
| MAIN | "allows compression but not out-of-line storage (... out-of-line storage will still be performed ... as a last resort)" | stays inline (the row fits a page), but every insert first tries to compress 2 kB of float16 noise; same page use as PLAIN |
| PLAIN | "prevents either compression or out-of-line storage" | inline; 3 rows per page (3 x 2,112 = 6,336 of 8,168 bytes; a fourth would need ≤ 2,042 bytes, impossible for 1,024 dimensions); 10 M / 3 x 8,192 = **27.3 GB**; exact scans and bulk reads are one sequential heap scan; pgvector's README gives `SET STORAGE PLAIN` for exactly this reason |

Decision: **PLAIN**. It costs 4.6 GB more disk per 10 M vectors (+20 %) and removes the per-row
chunk lookups from the two reads that matter most here: the exact search over a small filtered set
(plan 15 section 7) and the bulk read of a clustering job. Spaces of other dimensions use the same
template; at 768 dimensions or less a row fits four or more to a page under PLAIN anyway.

### 6.3 Adding and dropping a space

The status machine stays (`TRANSITIONS`, `src/domain/embedding_spaces.py:60-67`):

| Transition | Storage action (DDL by the job as `op_vector`, autocommit; the status change in its own transaction as `op_app`) |
|---|---|
| (insert) `pending` | `embedding_spaces` row; the job creates the table from the template, then checks the row-security flags |
| `pending` → `building` | the reindex job (`embedding_reindex_runs`, as `op_app`) fills it in batches of 2,000 through the staging pattern of section 1, deleting `vector_gaps` rows it fills; no index exists yet, so the fill never waits on one |
| `building` → `ready` | `CREATE INDEX CONCURRENTLY` from `vector_index`; then the job reads `pg_index.indisvalid` and refuses `ready` on an invalid index. A build that failed or was killed "leave[s] behind an invalid index" that still costs update overhead, and `IF NOT EXISTS` would skip it (CREATE INDEX page): the job drops it with `DROP INDEX CONCURRENTLY` before every retry. An expression index (1-bit codes, `binary_quantize(embedding)::bit(<dim>)`) waits, like every concurrent build, for older transactions to end, so the job runs it outside peak ingest. Then `role` may become `primary` (`uq_embedding_spaces_one_primary`) |
| `ready` → `retiring` → `retired` | readers stop at `retiring`; at `retired` the job drops the table in a short transaction with `SET LOCAL lock_timeout = '5s'`, retried, so a waiting `ACCESS EXCLUSIVE` never queues readers for long; the space's `vector_gaps` rows go by cascade |
| `failed` → `building` | the job truncates the table and drops any invalid index; the fill starts again |

`embedding_spaces` change: drop `index_role`, `vector_field`, `state_field`, `nested_path`,
`knn_method` and their two `CHECK`s; add `vector_table text NOT NULL UNIQUE` and
`vector_index jsonb NOT NULL` (a `VectorIndexSpec` pydantic body: method, operator class,
parameters; gate B owns its values). The `KnnMethodSpec` model and `physical_names` /
`legacy_names` (`embedding_spaces.py:70-135`) are deleted in the vector lane.

### 6.4 Reading every vector of a scope

```sql
COPY (
  SELECT v.subject_key, v.embedding
  FROM   vec.v<lock_key>_<space_key> v
  JOIN   item_state s ON s.item_key = v.subject_key
  WHERE  s.project_id = :p AND NOT s.excluded AND NOT s.class_validated   -- the job's pool filter
  ORDER  BY v.subject_key
) TO STDOUT (FORMAT BINARY);
```

`COPY TO` "can also copy the results of a SELECT query" and applies the table's `SELECT` policies
([COPY](https://www.postgresql.org/docs/18/sql-copy.html)); the binary format avoids text parsing of
1,024 numbers per row ("somewhat faster than the text and CSV formats") and is read by psycopg's
`cursor.copy()` with the pgvector adapter. Measured on the record so far: 69,630 rows per second
(plan 15 section 1). Writes: ingest inserts a commit group's vectors with one multi-row `INSERT`;
a reindex job uses the staging `COPY` then `INSERT ... SELECT` (section 1).

**Filtered search (M9; plan 15 section 7).** The filter fields live in `item_state` and
`cluster_memberships`, never in the vector row, so a filtered search is a post-filter join: pgvector
filters "after the index scan", `hnsw.ef_search` defaults to 40, and an iterative scan
(`hnsw.iterative_scan`, off by default) stops at `hnsw.max_scan_tuples`, default 20,000
([pgvector README](https://github.com/pgvector/pgvector), "Filtering", "Iterative Index Scans"). At
10 M a 0.3 % filter keeps 30,000 rows; a depth of 2,000 (`semantic_search_run.py`) would need about
670,000 visited tuples, so a plain scan silently returns a fraction. The search first counts the
filtered rows (capped at the upper threshold) and picks **exact** search below the lower threshold,
an **iterative scan** (`SET LOCAL hnsw.iterative_scan = relaxed_order`, raised
`hnsw.max_scan_tuples`) between, a **plain index scan** above. Thresholds, scan limit, the index on
each table and `embedding_spaces.vector_index` are gate B's, measured on this join (HNSW on
`halfvec`, on bit codes with rescore, or an outside index fed from the vector tables).

### 6.5 Sizes at 10 M

| Space | Rows | Heap (PLAIN) | Primary key | Approximate index (gate B, estimate) |
|---|---|---|---|---|
| item, 1,024 dims | 10 M | 27.3 GB | 10 M x 20 B / 0.9 ≈ 0.22 GB | HNSW on `halfvec`, m 16: about 2,056 + 2 x 16 x 6 B per element at about 3 per page ≈ 27 GB, built in `maintenance_work_mem` of that order ("hnsw graph no longer fits into maintenance_work_mem" slows the build, [pgvector](https://github.com/pgvector/pgvector); default 64 MB, [resource settings](https://www.postgresql.org/docs/18/runtime-config-resource.html)); on 1-bit codes `binary_quantize(embedding)::bit(1024)` with float16 rescore: 128 + about 200 B per element ≈ 3.5 GB |
| frame, 1,024 dims | 3 M | 8.2 GB | 0.07 GB | as above, x 0.3 |
| box, 1,024 dims | 5 M | 13.7 GB | 0.11 GB | as above, x 0.5 |

## 7. Transactions

All writes run under `project_tx` (02a 1.4) at READ COMMITTED, from about 30 API processes and the
worker containers. **Lock order:** (1) the operation-like row being written (`label_operations`,
`erasure_jobs`, the operation being undone); (2) the `class_registries` row; (3) imohash advisory
locks, sorted (`lock_imohashes`, `src/db/repos/image_contents.py:71-89`); (4) `entities`; (5)
`photos` `FOR NO KEY UPDATE`; (6) `items` (`SELECT ... WHERE item_key = ANY(:keys) ORDER BY item_key
FOR UPDATE`); (7) `faces`; rows of one rank in ascending key order. A writer may skip ranks, never go
back, so writers wait but never deadlock ("acquire locks on multiple objects in a consistent order",
[explicit locking](https://www.postgresql.org/docs/18/explicit-locking.html)); `FOR NO KEY UPDATE`
does not conflict with the `FOR KEY SHARE` of child-row foreign-key checks (same page). A fold reads
its inputs after it holds the `items` lock, so a second writer waits and then folds the first one's
committed inputs (02a 1.3). **Zero rows** from a lock mean the row was deleted meanwhile: a
single-item human act re-reads by `crop_id` and retries, or answers 404 `item_not_found` and writes
nothing; a chunk skips and counts; a machine writer skips (the ledger keeps the photo listed); a
writer that needs a lower rank rolls the chunk back and retries with it taken first. A deadlock
(`40P01`, only through a bug in this order) is retried whole up to three times.

| Writer | First lock | Then | Zero rows |
|---|---|---|---|
| Ingest commit group (13a L4) | (3) imohash locks of the group's files, sorted (class creation runs before, 13a barrier B3) | probe `photos` by `image_id` (3.4); confirm by SHA-256 on a match, an id from SHA-256 on a collision (`content_key.py:8-13`); insert photos, items, detections, opinions, capture, ledger, vectors, state rows with their deltas, events | n/a: new rows; a duplicate is reported as today |
| Human act on existing items (label, discard, exclude, restore, dismiss, flag, move, resolve, accept machine, geometry, box edit, region verdict) | (1) its new op row | (6) items; box edit compares `region_revision` with `expected_region_revision` (409 `region_revision_conflict`) and bumps `box_seq`; insert claims, fold, update state, events | re-read by `crop_id`, else 404 |
| Bulk human act (batch label, bulk accept, holdout freeze, audit start, batch box state) | (1) op row (`writing`) | chunks of 1,000: (6) items, claims `ON CONFLICT DO NOTHING` (M4), fold, one `UPDATE ... FROM (VALUES ...)`; op `written` or `failed` with the count | skip, count |
| Undo of an operation (3.5) | (1) the undone op row, `FOR UPDATE`, set `undoing` | chunks: (6) items, withdraws, fold, in one transaction; `undone` after the last | withdraws still written; nothing to fold |
| Draw an object (`item_drawn`) | (1) op row | (5) the photo; insert item; claim; fold | photo erased: 404 |
| Region stage, VLM, auto-label, open-vocab target, assist answers that change the class | none | (6) the batch's items; replace `region_boxes` and `box_texts` (old ids by IoU, new ids from `box_seq`), bump `region_seq`/`geometry_seq`; upsert results; fold | skip the item |
| Probe, scores, projections, text lines | none | upsert result rows only (no fold layer reads them); the FK check's `FOR KEY SHARE` waits for a deleting attach and then raises `23503`, which skips the row | skip the row |
| Vector writes (ingest, reindex, box vector refresh after a box edit) | none | (5)/(6)/(7) `FOR KEY SHARE` on the subjects' identity rows, the lock an FK check would take; then insert or replace vectors, delete gaps | skip the subject |
| Attach: re-detect, open-vocab run, import reconcile, re-ingest | (5) the photo | (6) its items; `claim_match`; insert or delete `items`; detections; fold | n/a (holds the photo) |
| Adopt detector, assist-loop propagate | (1) op row | chunks: (6) items; insert opinions with `op_id`; fold. Undo: chunks delete the op's opinions and fold | skip |
| Dataset import, combine (per image group) | (1) the import's op row | class creation before (2); (3) imohash locks for new photos; (5) existing photos; (6) items; claims, opinions, detections, notes with `import_id`; fold; `fold.conflicts` gives `locked_conflict` / `combine_conflict` | skip, count |
| Near-duplicate run activation; duplicate verdict | the run row (OCC on `near_dup_runs`, 09d) | a verdict is one claim with no fold of items; a holdout closure is a bulk human act | n/a |
| Cluster activation | the `cluster_activations` row (OCC: `revision = :expected`, 02d 2.2) | nothing: readers join the serving partition (M6) | 409 on a stale revision |
| Class create, rename | (2) registry row | `project_classes` row | n/a |
| Class merge (M16) | (2) registry row | `project_classes.merged_into`; commit. Lists and counts are right from this commit because readers filter `class_id = ANY(family)` and group by the merge root. Then a job: chunks of (6) items with the source class through `apply()` with no new input (the fold resolves `merged_into`; `store_states` emits the deltas, 15b 6.2); a fold racing it resolves the same way; at the end the family drops the source id (15b P1). A source class with frozen holdout items is refused (409, `classes_merge.py:74`) | skip |
| `fold.rebuild` (M11) | the job lease | chunks of (6) items in key order; `state.store_states` writes differing state rows, deletes rows without identity and emits their deltas | skip |
| Summary compactor (15b 6.5) | the job lease | one statement: `DELETE` a batch of `sum_deltas` (`FOR UPDATE SKIP LOCKED`) `RETURNING` it, upsert the sums into the base rows | locked deltas skipped |
| Summary rebuild, partition or run fill (15b 6.6, 6.7) | the job lease (fill: after the `cluster_activations` OCC write) | one `REPEATABLE READ` snapshot; insert correcting deltas; no lock | n/a |
| Photo or media claims (phase 5) | (1) op row | (5) the photo; claim; `fold_photo`; `photo_state` | 404 |
| Entity and face claims (phase 5) | (1) op row | (4) entities, (7) faces; claims; `fold_entity`; `entity_state`, `membership_state`, `face_state` | 404 or skip |
| Erasure (section 8), project delete | (1) the `erasure_jobs` row | (4) entity, (5) photos `FOR UPDATE` in key order, (6) their items; `state.store_states(delete_photos=...)` (deltas, 15b 6.2); catalogue deletes | already erased: skip |
| Single-label op pruning (3.5) | none | `DELETE` of old `label_operations` rows; claims untouched | n/a |

No path locks `items` and then a `photos` row, and no path writes `item_state` or `box_state` rows
without first holding the matching `items` lock, so the order is total. Summary upkeep adds no
rank: writers only insert `sum_deltas` rows with fresh keys, which wait on nothing (15b 6.3), and
the compactor is the only writer of base summary rows.

## 8. Erasure

The one path that deletes claims (plan 15 R3, section 14 answer 4; B5). `src/services/erasure/` runs
one job per subject: a **photo** (also emptying the trash), a **person** ("forget this person",
photo_library 15 section 2.1), a **project** (today's `project_delete`, 02a 1.2). It inserts an
`erasure_jobs` row, sets the transaction-local guard `op.erasure_job` in each chunk, removes the
chunk's counted rows through `state.store_states(delete_photos=...)` so every summary loses them in
the same transaction (15b 6.2; a cascade alone would leave the summaries high), deletes through
the catalogue, deletes the files, writes a counts-only `privacy_purges` row, and at `done` nulls the
subject columns.

```sql
CREATE TABLE erasure_jobs (
  id uuid PRIMARY KEY DEFAULT uuidv7(), PROJECT,
  subject_kind text NOT NULL CHECK (subject_kind IN ('photo','person','project')),
  photo_keys bigint[] NULL, entity_key bigint NULL,   -- the subject; nulled at done
  state text NOT NULL CHECK (state IN ('running','done','failed')),
  counts jsonb NOT NULL DEFAULT '{}', job_id uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(), finished_at timestamptz NULL
);
-- op_erasure_allows(p): current_setting('op.erasure_job', true) names a 'running' erasure_jobs row of project p.
-- claims_append_only and 02d's cluster_children_ready_immutable allow DELETE when it is true (a cascade from photos
-- fires them too); EXECUTE granted to op_app; a test fails if a module outside src/services/erasure/ sets
-- op.erasure_job (the op_project_delete_allows pattern, 02a 1.2).
```

**Catalogue** (`src/services/erasure/catalogue.py`): every (table, key column) that can hold a subject.

| Reached from a photo | How |
|---|---|
| `photos` | the root delete |
| `photo_provenance`, `analysis_ledger`, `photo_capture`, `items`, `detections`, `text_lines`, `faces`, `claims` (`photo_key`), `photo_state`, `photo_files`, `photo_derivatives`, `photo_signals`, `photo_places` | `ON DELETE CASCADE` on `photo_key` |
| `class_opinions`, `item_region_results`, `region_boxes`, `box_texts`, `probe_predictions`, `item_scores`, `item_projections`, `assist_answers`, `item_notes`, `item_state`, `box_state`, `face_state` | cascade through `items` / `faces` |
| vector tables (`embedding_spaces.vector_table`), by `target` | `DELETE ... WHERE subject_key = ANY(photo, item, box and face keys)`, before the photo delete |
| `vector_gaps`, `cluster_memberships`, `near_dup_members`, `near_dup_edges`, `holdout_near_dup_moves`, `entity_edges`, `membership_state` | explicit delete by subject key (no foreign key by design) |
| `claims` that name the photo without `photo_key` (`subject_pair_id`, or `subject_crop_id` of a pair) | explicit delete by wire id |
| `dataset_import_images.photo_key`, `entity_state.cover_photo_key` | `SET NULL` |
| `job_items.item_id`, `events.payload`, `jobs.params`, `label_operations.params` | explicit delete of rows or entries that name the wire ids; events beyond retention are already gone |
| files | the original and bundle members when the project owns them (uploads; a source root read in place is never written, plan 15 section 8), crop and thumbnail cache entries, exports (`dataset_exports`, `export_support.py:435`), the holdout snapshot (`holdout_freezes.snapshot_storage_key`, `holdout.py:285`: rewritten without the photo), training data copies under each run's `artifact_prefix` |
| **not erased** | trained weights (`models`, `model_engines`, run checkpoints): a model trained on the photo keeps what it learned; owner question (plan 15 section 13) |

A **person**: its `entities` row and what cascades from it (`entity_state`, `entity_edges`,
`membership_state`, claims by `subject_entity_key`), the faces assigned or merged to it with their
vectors, `face_state` and claims (`subject_face_id`), exclusion rules and memories about it, cached
face crops; the photos stay. A **project**: today's batched delete plus `DROP TABLE` of its `vec`
tables. **Tests**: `test_erasure_catalogue_complete` fails when a table with a column named
`photo_key`, `item_key`, `box_key`, `face_key`, `entity_key`, `subject_key`, `subject_*_id`,
`image_id`, `crop_id`, `face_id` or `member_id`, or a vector table named by `embedding_spaces`, is
missing from the catalogue; `test_erasure_leaves_nothing` erases a seeded photo and person and finds
no row, no file and no nearest-neighbour hit on a deleted vector (photo_library 15 exit criterion).

## 9. Migration

Current head: `0009_ingest_retry_at` (`src/db/migrations/versions/0009_ingest_retry_at.py:23`).
Revision ids are fixed here and passed with `--rev-id`; one revision-writing lane per step (02d
section 0).

| Revision | Lane (plan 15 section 10) | Creates or changes |
|---|---|---|
| `0010_storage_record` | step 1 | `CREATE EXTENSION IF NOT EXISTS vector` and `pg_trgm` (m5: the image is the pgvector project's `pgvector/pgvector:0.8.7-pg18-trixie`, pinned in 01 (decided in the database lane, round 3); `pg_trgm` is a trusted contrib module, [pg_trgm](https://www.postgresql.org/docs/18/pgtrgm.html); a test runs both statements as `op_owner` on the real image, and if pgvector needs more, `db-migrate`'s bootstrap step creates it as `op_super`); schema `vec` with `CREATE` for `op_vector` only; every table of sections 2 and 8 with its keys, foreign-key indexes, policies and triggers (`claims_append_only`, `analysers_immutable`, the erasure guard `op_erasure_allows`); the `claim_kinds` rows; the `label_operations` changes; `dataset_imports.dataset_name`; the `embedding_spaces`, `cluster_memberships`, `near_dup_*`, `holdout_near_dup_moves`, `dataset_import_images` changes; grants to `op_app`. The role `op_vector` is created by `db-migrate`'s bootstrap with the other roles (02a 1.5), from its own secret file. Existing space rows are reset to `pending`, so the embedding-space job creates their tables (data is re-ingested). The indexes of section 2.6 come in the revision the physical-design pass writes |
| `0011_ingest_cutover` | step 2 | drops `image_contents` |
| `0012_claims_cutover` | step 3 | drops `label_operation_members`, `near_dup_decisions` |
| `0013_results_cutover` | step 4 | drops `search_projection_tasks` |
| `0014_reads_cutover` | step 5, after gate A | drops `project_indexes` (one store) or creates `item_changes` (search copy) |

Creation order inside `0010`: `analysers`, `claim_kinds`; `photos`, `photo_provenance`; `items`;
`entities` (claims reference it); `claims`; the result tables; `item_state`, `box_state`; the rest of
2.5; `vector_gaps`; `erasure_jobs`.

No data migration: the 0.7 data is wiped and re-ingested (owner, 2026-11-14); `downgrade()` of each
revision drops what it created (02d section 0). A test `tests/db/test_storage_schema_matches_plan.py`
introspects the migrated database and asserts every table, unique constraint, partial index,
trigger and policy named here, and the row-security flags of every `vec` table, as 02a's schema
test does; `src/db/table_use.py` gets a row per new
table (writer, reader, wave), with the phase-5 tables marked "schema-test assertion only" until
phase 5 (02d section 0.1 rule).

## 10. Contract diff

Plan 15 section 11: the wire contract is unchanged except these fields (M19). Every other field of
`wire.py:287-460` and the list filters derives from the tables above (section 5); the lanes run
`make contracts` and commit `contracts/` with each change.

| Wire field (model) | Change | Reason |
|---|---|---|
| `image_path` (item and image models) | the value becomes the storage display path `<root>/<relative path>`, rendered from `photos.storage_key` by one `RootedStore` function, instead of a host path | R9: no stored value is a host path |
| `source_image_path` (item) | deleted | always equal to `image_path` (`wire.py:322-323`) |
| `conflicts`, `conflict_ids` (undo responses, `src/schemas/curation/label_ops.py:77-78`) | deleted | claim writes wait on row locks and never fail on an optimistic conflict (section 7); the count would always be zero |
| `kept_changed` (undo responses) | an item excluded after the operation is no longer kept: its claims are withdrawn and it counts `restored` | an operation is undone whole (3.5); exclusion no longer holds a class in place (M2) |
| `class_unresolved` (item) | added, boolean | a claim whose class name no longer resolves keeps the item locked with no class (M17); the screen must say why |
| `class_source` of a combine label | `external_label`, as an import | layer 2 |
| `region_revision`, `geometry_revision` (item) | same meaning (a strictly increasing token, the thumbnail `?v=`); the values restart at re-ingest and also count machine result writes | 4.3 layer 6 (M11) |
| `frame_mismatch` (reprocess and import job results) | added, count of claim groups left dormant because the photo's decode frame changed | 3.4 step 1 (M14) |

## 11. Questions for the coordinator

Revision 0's six questions are answered in plan 15 section 14 and applied here: the reserved core
project (section 1), the `op_vector` role (6.1), gallery claims (2.5), erasure (8), the
machine-class order (4.2), identity apart from state (4.1). New:

1. **A person's label on a frozen holdout item.** No route refuses it today (`crops_write.py:39`
   checks only that the item exists); R3 now says "while frozen in the holdout, its class is the frozen
   one". This file reads that as binding machine writers and undo only, so a person's newer claim
   still decides (4.3 layer 2). If a person must not relabel a frozen item either, the label routes
   gain a 409 and layer 2 ignores claims newer than the freeze.
2. **Machine boxes that arrive after a person closed an item's box set** stay results and are not
   shown (4.3 layer 4), which is today's effect of the item lock (`occ_locks.py:137-151`) expressed
   as data. A re-run of the region stage therefore cannot add a box to a validated item; confirm, or
   name a route that reopens the set.
3. **Trained weights after erasure** are not erased (section 8); the owner decides whether such a
   model must be retrained or deleted (plan 15 section 13).

Answers (coordinator decisions 2026-11-15, applied by the fold lane; the rows are in appendix A,
"Fold lane decisions"): question 1 stands as written (a person's newer claim decides a frozen
item; an import or combine claim after the freeze does not, Q3); question 2 stands, now exact: the
generation the person saw stays and a re-run's boxes stay results (Q9).

## Appendix A. Review trace

One row per finding of the independent review (2026-11-15) and the section that now answers it.

| Finding | Answered in |
|---|---|
| B1 validated box not pinned | plan 15 R3; 4.3 layer 1 (pin to `subject_bbox`); `claim_kinds.pins_geometry` (3.2); test `test_pin_survives_machine_writes` |
| B2 import overrides a human | plan 15 R3; 4.3 ranking; 3.5 Import (`locked_conflict` from `fold.conflicts`); test `test_human_outranks_import_and_combine` |
| B3 whole-set region verdicts | 3.2 (`region_status_set` only `no_region_visible`, `box_deleted` dormant); 3.3 expansion; 4.3 layer 4 (closed set, `region_validated`) |
| B4 undo dies between withdraw and fold | 3.5 Undo (chunk-atomic, no operation withdraw, stored status); 2.2; section 7 undo row; test `test_undo_chunks_atomic` |
| B5 erasure reach | section 8 (catalogue, `erasure_jobs` guard, files, schema test); plan 15 section 14 answer 4 |
| M1 holdout not frozen without a machine class | 3.3 `holdout_frozen` payload; 4.3 layer 2 holdout rule; 3.5 `kept_holdout` |
| M2 excluded items stay validated | 4.3 layers 2 and 3; 5.2 `excluded_prior_class_validated` |
| M3 imohash lock and insert order | section 7 rank 3 and ingest row; 2.1 lookup note |
| M4 bulk chunks without idempotency key | 2.2 `uq_claims_op_subject`; 3.5 Operations and step-back undo |
| M5 `item_state` never HOT, no autovacuum story | 2.4 (narrow row, `IS DISTINCT FROM`, `updated_at`, autovacuum settings, HOT target); 2.6 |
| M6 cluster activation rewrites every row | 2.3 `cluster_memberships` as the placement; section 7 activation row; plan 15 section 13 |
| M7 index and sort claims | 2.6 (rules and one row per read, from gate A) |
| M8 label-op current counts | 2.4 `class_op_id` and its partial index |
| M9 filtered vector search recall | 6.4 three search paths; plan 15 section 7 |
| M10 runtime DDL role | 6.1 `op_vector`; 6.3 `indisvalid`, retries; plan 15 section 14 answer 2 |
| M11 rebuild undefined, `region_revision` not reproducible | 4.1 chunked rebuild under lease; 2.3 `items.geometry_seq`, `region_seq`; 4.3 layer 6 |
| M12 undo outcomes and discard granularity | 3.5 outcome table and step-back undo; section 10 |
| M13 adopt, propagate, suggestion imports | plan 15 R2; 2.3 not-rebuildable rows, analyser kind `import`, `import_id`; 3.2; 3.5 Import; 4.3 layer 3 `import_ids` |
| M14 attach per claim, frame | 3.4 (groups, detection choice, item rows, split); 2.1 `frame`; 2.2 `subject_frame` |
| M15 writers missing from the concurrency table | section 7 writer table and zero-rows rule |
| M16 class merge filters and counts | section 7 class merge row; 2.6 class family |
| M17 cleared, empty VLM answer, unresolved class | 4.3 layer 2; 4.2; 2.3 `class_opinions` upsert; 3.3 resolution |
| M18 gallery tables | 2.2 `claim_kinds`, subject `media`; 2.5 reshaped tables; 3.2 phase-5 kinds |
| M19 wire fields not derivable | 3.3 `labeler`, `profile`; 2.3 `item_notes`, probe confidence, duplicate-group paths; 5.2; section 10 |
| M20 hint-first class resolution | 3.3 class resolution |
| M21 stale `class_moved` placement | 4.3 layer 5; test `test_old_move_not_placed` |
| m1 claim size | 2.2 sizes (payload re-estimated: 6.1 GB) |
| m2 ledger "below a version" | 2.3 `analysis_ledger` equality rule |
| m3 duplicate `crop_id` unique key | 2.4 (no unique copy on `item_state`) |
| m4 single-label operation growth | 3.5 Operations (pruning); 2.2 `op_id` without foreign key |
| m5 pgvector in the image | section 9, `0010` row |
| m6 provenance key with a nullable job id | 2.1 `uq_photo_provenance_source` |
| m7 two work lists | 2.3 `analysis_ledger` is the only work list; `vector_gaps` explains only |

### Fold lane decisions

Decisions of the coordinator on 2026-11-15 after the fold lane's differential run against today's
code (`tests/domain/test_fold_differential.py`); each is applied in `src/domain/` and in the
allow-list `tests/domain/world_allowed.py`.

| Decision | Rule now | Answered in |
|---|---|---|
| Q1 `needs_new_class` | a later human `class_set`, a plain label too, clears the flag (today only resolve does) | 4.3 layer 3; test `test_a_plain_label_clears_needs_new_class_q1` |
| Q2 audit outcome | the latest human class verdict after `audit_sampled`, as today | 4.3 layer 3 |
| Q3 import on a held-out item | an import or combine class claim after the freeze does not decide, unless it corrects the import's own class; the writer still reports the conflict | 3.5 Import; 4.3 layer 2 |
| Q4 step-back undo | does not step back over a bulk accept, as today | 3.5 Undo |
| Q5 dormant claim group | step 3 of matching also tries the `item_bbox` the group's newest claim saw | 3.3, 3.4 step 3; test `test_claim_match` |
| Q6 person-made box id | a created box owns its minted id and matches on it only; claims on machine boxes match on id, then IoU | 3.4 step 7 |
| Q7 whole-set verdict over no box | keeps the status given (`verify_rejected`, `false_positive`); `no_region_visible` only when given | 3.2, 3.3, 4.3 layer 4; test `test_a_verdict_over_no_box_keeps_the_status_the_person_gave_q7` |
| Q8 `region_set_complete` | a fold output and an `item_state` column | 2.4; 4.3 layer 4 |
| Q9 `region_seen_seq` | a fold output the repository stores; a claim's `region_seq` is the generation the person saw | 2.4; 3.3; 4.3 layer 4 |
| Q10 the clear | a `class_cleared` payload lists the hidden (source, analyser) pairs; an opinion is hidden iff its pair is in a live clear's list | 3.3; 4.2; 4.3 layer 2; tests in 4.5 |
| Q11 what pins the box | exactly while today's `is_locked_item` holds; a clear, dismissals, flags, a move and another exclusion reason do not pin | 3.2; 4.3 layer 1; 4.4; tests in 4.5 |
| Q12 text consensus | the fold takes the resolved reader consensus (`MachineText`) as input; the wiring lane moves `region_text.py` consensus into `src/domain/` | 4.3 layer 4 |
| Q13 class profile | `profile` is `{detector?, detector_version?}` | 3.3 |
| Q14 phase-5 kinds | not in the catalog until the gallery lane adds them | 3.2 |

### Database lane decisions

Decisions of the database fix lane on 2026-11-15, after the first server run of `0010_storage_record`
failed at `CREATE SCHEMA vec` (`permission denied for database`).

| Decision | Rule now | Answered in |
|---|---|---|
| Placement of the privileged steps | the owner role has no `CREATE` on the database, so the superuser bootstrap (`src/db/roles.py`: `ensure_roles`, `prepare_database`) creates the role `op_vector` (secret `OP_DB_VECTOR_PASSWORD_FILE`), installs `pg_trgm` and, when the image offers it, `vector`, and creates the schema `vec` owned by `op_owner`; `apply_grants` (as the owner) gives `op_vector` `USAGE, CREATE` on `vec` and `REFERENCES (id)` on `projects`; the migration creates none of them, checks them, and fails naming the bootstrap step | 6.1; 12 `0010_storage_record` step 1; 02a 1.5 |
| `vec` outlives a downgrade | the schema belongs to the bootstrap, so `downgrade()` no longer drops it | 6.1 |

### Reads and slices review

The review of 15b and 15c (2026-11-15) with the coordinator's decisions; one row per finding.

| Finding | Answered in |
|---|---|
| B1 `image_contents` used by core ingest | 2.6 migration paragraph (`0012` fences it, no drop); 15c S1a-ii, section 8 |
| B2 exclusive rebuild lock | 2.6 summary paragraph; section 7 (rebuild and fill rows, no lock); 15b 6.6, 6.7 |
| B3 summary rows held to commit | 2.6 summary paragraph (no rank 8); section 7 closing paragraph; 15b 6.3 |
| M1 class family on X2 | 2.6 X2; 15b P1 |
| M2 drained queue | 2.6 X2 (`class_validated` key), X4 (review partial index); 15b section 5 |
| M3 index per sort | 2.3 `items.sort_area`, `sort_rank`; 2.6 X28 to X32; 15b section 7 |
| M4 drift | 2.4 write rules; section 7 class merge, `fold.rebuild`, erasure rows; section 8; 15b 6.2 |
| M5 capped counts as guards | 15b section 5 |
| M6 box rows | 2.6 X26, X27; 15b section 9 |
| M7 dropped reader behaviours | 2.6 X13 on `taken_local`; 15b P1, P15, sections 7 and 8 |
| M8 attach owner | 15c S1a-ii |
| M9 deletion order | 15c section 8 |
| M10 red by file | 15c section 9 |
| M11 slice size | 15c sections 1 and 3 |
| M12 round trips per label | 15b 6.9, section 13 M1 |
| m1 P6 short page | 15b P6 |
| m2 X22 direction | 2.6 X22 |
| m3 upsert order | 15b 6.5 |
| m4 advisory key space | section 7 (no tally lock remains) |
| m5 P4 heap fetches | 15b section 5 rule 8 |
| m6 machine churn | 2.4 `updated_at` rule; 2.6 HOT paragraph |
| m7 GIN `fastupdate` | 15b section 8; 15c S4, S5 |
| m8 one filter file | 15b P1 |
| m9 `photo_counts` per photo | 15b 6.1, 6.2 |
| m10 mixed-direction sorts | 15b sections 4 and 7 |
| m11 capped-total classification | 15b section 5 |
