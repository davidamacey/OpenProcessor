# 15a: storage schema, the tables of the record

| Field | Value |
|---|---|
| Status | plan, revision 0 (2026-11-15). Companion of [15_storage_plan.md](15_storage_plan.md) section 12; its decisions are fixed here, not reopened |
| Code base read | `lane/speed-plan` at `ca918ea2d` (integration code of `int/b4-stateless` plus plan 15); the storage inventory of 2026-11-13 (`artifacts_local/speed_1113/storage_inventory.md`, gitignored), re-checked in code where a design depends on it |
| Conventions | [02a_postgres_schema.md](02a_postgres_schema.md) section 1 and [02d_research_seam_schema.md](02d_research_seam_schema.md) section 0.1 apply to every table here, except where section 1 below says why not |
| Server | PostgreSQL 18 (`postgres:18.6-trixie`, [01_package_pins.md](01_package_pins.md)); pgvector 0.8.7, the newest release (not pinned in 01; the lane adds the pin, [pgvector README](https://github.com/pgvector/pgvector)) |
| Not here | the query map (15b), the lane specs (15c), what the models compute |
| Data | wiped and re-ingested (owner, 2026-11-14): no migration of search-index documents |

Plain words used throughout: a **photo** is a stored image (today the `images` document); an **item**
is a detected or drawn object (today the `items` document, wire id `crop_id`); a **box** is one region
inside an item (today one element of `region_boxes`); a **claim** is one human decision; a
**result** is one machine output; the **fold** turns results and live claims into the current state.

## 0. What this file fixes

It gives every table of the record with columns, keys, indexes and row-level security; maps every
field stored today to a column, a value derived at read, or a deletion; defines the claim kinds and
their payloads; defines the fold and proves today's lock rule from the shape of the data.

| 02a / 02d table | Fate | Section |
|---|---|---|
| `image_contents` | **deleted**: merged into `photos` (plan 15 section 4.1); dropped by the ingest lane | 2.1 |
| `label_operations` | **changed**: groups claims; `undone_at` goes (an undo is a claim) | 2.2 |
| `label_operation_members` | **deleted**: claims carry their operation; no prior state is stored | 2.2 |
| `near_dup_decisions` | **deleted**: a duplicate verdict is a claim | 3 |
| `search_projection_tasks` | **deleted**: no projection into documents remains (plan 15 section 9) | 9 |
| `embedding_spaces` | **changed**: the OpenSearch names (`index_role`, `vector_field`, `state_field`, `nested_path`, `knn_method`) give way to `vector_table` and `vector_index` | 6 |
| `cluster_memberships` | **changed**: `item_key text` becomes `subject_key bigint` | 2.3 |
| `near_dup_members`, `near_dup_edges`, `holdout_near_dup_moves` | **changed**: text member ids become `bigint` subject keys | 2.3 |
| `dataset_imports` | **changed**: gains `dataset_name` (today `import_dataset_name` sits on every item) | 2.1 |
| `dataset_import_images` | **changed**: `image_id text` becomes `photo_key bigint`, `item_ids` goes (items of an import are its claims) | 2.1 |
| `project_indexes` | kept until gate A; deleted with the search engine if gate A ends in "one store" | 8 |
| every other 02a / 02d table (`projects`, `class_registries`, `project_classes`, `holdout_freezes`, `cluster_partitions`, `cluster_identities`, `partition_clusters`, `cluster_activations`, `near_dup_runs`, `near_dup_groups`, `near_dup_pending`, `embedding_model_versions`, `jobs`, `events`, ...) | **kept** as written | |

New tables: 28 created now (section 2), of which the 6 gallery tables of section 2.5 are filled
first in phase 5; plus one vector table per embedding space, created at run time from one template
(section 6). Changed: 8. Deleted: 4, and `project_indexes` if gate A ends in one store.

## 1. Conventions used here

| Convention | Rule | Tie to 02a, and where it cannot hold |
|---|---|---|
| Keys inside | high-volume rows (photos, items, boxes, faces, detections, claims) have `bigint GENERATED ALWAYS AS IDENTITY` keys ([identity columns](https://www.postgresql.org/docs/18/ddl-identity-columns.html)); low-volume entity rows keep `uuid DEFAULT uuidv7()` | 02a 1.1 says entity keys are `uuidv7()`. At 10 million items a `uuid` costs 16 bytes in the row and in every index and child row, against 8; the locality argument of 02a (inserts stay at the right edge of the B-tree) holds for an identity column too. R8 of plan 15 asks for compact integers. This is the one 02a convention this file changes |
| Wire ids | content-derived 128-bit ids (`image_id` = `sha256(imohash)[:32]`, `src/services/near_duplicates/content_key.py:94`; `crop_id` = `sha256(image_id\|bbox)[:32]`, `src/services/detection/geometry.py:61`; drawn `crop_id`, `src/domain/item_geometry.py:75`) are stored in a `uuid` column, unique per project beside the integer key. PostgreSQL accepts 32 hex digits without hyphens as `uuid` input ([UUID type](https://www.postgresql.org/docs/18/datatype-uuid.html)); one helper `src/domain/wire_ids.py` renders the value back as 32 lowercase hex, so the wire is unchanged. 16 bytes instead of 33 for `text` | new; 02a has no wire-id column of this size |
| Box ids | `box_id text` `b<N>`, minted from `items.box_seq` (today `region_box_seq`, `src/domain/region_boxes.py:133`) | unchanged rule |
| Project column | every table that holds project data has `project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE` and the policy below; repositories always add `project_id = :p` (02a 1.4, RC-9) | 02a 1.2, 1.4. Child tables keep a direct `project_id` instead of the `EXISTS` policy of 02a section 5, because they are unbounded (the X-1 rule of 02d section 3) |
| Policy | `ENABLE` and `FORCE ROW LEVEL SECURITY`, policy `p_<t>_project` exactly as 02a 1.4 (the InitPlan form of `op_scope_*`), called **RLS(02a)**. In the DDL below `PROJECT` stands for `project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE`, and **every table with a `project_id` column has RLS(02a)**; only `analysers` (a deployment catalog) has none | 02a 1.4; the test `tests/db/test_row_level_security.py` already iterates every `project_id` table |
| No `project_id IS NULL` | none of the new tables allows a NULL project; the core API's data gets a project (question 1) | 02a 1.4 allows NULL rows under `core` scope for a few tables; a nullable column in every key at 10 million rows is not worth it |
| Names | tables plural snake_case; constraint names from the `MetaData` naming convention (02a 1.1); every closed set is `text` + named `CHECK` with one Python source and a test | 02a 1.1, RR-17 |
| Times | `timestamptz NOT NULL DEFAULT now()`; capture times stay floating `timestamp` (the camera's local wall clock), as the `capture` mapping stores them today (`src/clients/curation_opensearch/capture_fields.py:35`) | 02a 1.1 |
| Actors | `*_by text` as 02a 1.1 (`api`, `worker:<host>`, ...); claims add `actor_kind` | 02a 1.1 |
| `jsonb` | only for bodies validated by a pydantic model on write and read: claim payloads, assist answers, analyser parameters | 02a 1.1, D-59 |
| No partitioning | no table here is partitioned. Partition by project would make `project_delete` a `DROP`, but creating a partition is DDL and `op_app` has none (02a 1.5); the planner handles "up to a few thousand partitions fairly well" only with pruning ([partitioning](https://www.postgresql.org/docs/18/ddl-partitioning.html)). The batched delete of 02a 1.2 stays | 02a 1.2, 1.5 |
| Bulk writes | `COPY FROM` is refused on a table with row-level security ("Currently, COPY FROM is not supported for tables with row-level security", [COPY](https://www.postgresql.org/docs/18/sql-copy.html)). Bulk writers use the pattern already in `src/services/clustering/partitions.py:30-44`: `COPY` into an `ON COMMIT DROP` temporary table, then one `INSERT ... SELECT` that the policy checks | the 02d note "bulk-written with COPY" means this pattern |
| Updated rows | a table whose rows are updated in place gets `fillfactor` below 100 so updates can be HOT: a HOT update needs "no indexed column modified" and "sufficient free space on the page" ([HOT](https://www.postgresql.org/docs/18/storage-hot.html); fillfactor 10-100, default 100, [CREATE TABLE](https://www.postgresql.org/docs/18/sql-createtable.html)) | new |

Size arithmetic: an 8 kB page holds 8,168 bytes of tuples; a tuple costs a 24-byte header (23
rounded), its columns with alignment, and a 4-byte line pointer. A B-tree entry costs about 8 bytes
of header plus the key, padded to 8, plus page fill of about 90 %. "10 M" below means a project with
10 million items, about 3 million photos (3.3 items per photo, the set P ratio of plan 13), 5 million
boxes and 20 million claims; each figure shows its arithmetic.

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
CREATE INDEX ix_photos_imohash ON photos (project_id, imohash);                 -- the ingest duplicate check (RD-43)
CREATE UNIQUE INDEX uq_photos_sha256 ON photos (project_id, content_sha256) WHERE content_sha256 IS NOT NULL;
CREATE INDEX ix_photos_source_identifier ON photos (project_id, source_identifier) WHERE source_identifier IS NOT NULL;  -- shape 31a
CREATE INDEX ix_photos_created ON photos (project_id, created_at, photo_key);    -- shape 24 sort 'indexed', shape 31d
-- Rows are never updated except content_sha256 (once); fillfactor 100.
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
CREATE UNIQUE INDEX uq_photo_provenance_source ON photo_provenance (photo_key, kind, coalesce(import_id, combine_job_id));
CREATE INDEX ix_photo_provenance_import ON photo_provenance (project_id, import_id) WHERE import_id IS NOT NULL;
```

`dataset_imports` gains `dataset_name text NULL`; `dataset_import_images.image_id text` becomes
`photo_key bigint NULL REFERENCES photos ON DELETE SET NULL` and `item_ids` is dropped.

Sizes at 10 M (3 M photos): `photos` tuple 24 + 8 + 16 + 16 + 17 + 33 (storage key, about 60
characters) ... about 220 bytes with text, so 3 M x 224 / 8,168 = 82 k pages, **0.7 GB**; four
indexes about 3 M x 40 B = 0.12 GB each, **0.5 GB**. `photo_provenance` only for imported or combined
photos: at most 3 M x 120 B = 0.36 GB.

### 2.2 Decided: claims and operations

The claim log is section 3. Its DDL:

```sql
CREATE TABLE claims (                        -- append only: no UPDATE, no DELETE (trigger claims_append_only)
  claim_id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,   -- the rising id; claim order
  at                  timestamptz NOT NULL DEFAULT now(),
  actor_kind          text NOT NULL CONSTRAINT ck_claims_actor_kind CHECK (actor_kind IN ('human','import','combine')),
  actor               text NOT NULL,         -- 'api', 'import:<id>', 'combine:<job>' (02a 1.1)
  actor_user_id       uuid NULL,             -- v0.8.0 authentication; no FK now (02a 1.1)
  op_id               uuid NULL REFERENCES label_operations(id) ON DELETE NO ACTION,
  kind                text NOT NULL CONSTRAINT ck_claims_kind CHECK (kind IN (/* section 3.2, from src/domain/claim_kinds.py */)),
  subject_kind        text NOT NULL CONSTRAINT ck_claims_subject_kind
                      CHECK (subject_kind IN ('photo','item','box','pair','face','entity','operation','claim')),
  photo_key           bigint NULL REFERENCES photos(photo_key) ON DELETE CASCADE,
  subject_crop_id     uuid NULL,             -- the item's wire id when the claim was made
  subject_box_id      text NULL,
  subject_bbox        real[] NULL CHECK (subject_bbox IS NULL OR cardinality(subject_bbox) = 4),          -- item or box geometry at the time
  subject_machine_bbox real[] NULL CHECK (subject_machine_bbox IS NULL OR cardinality(subject_machine_bbox) = 4), -- the detector box at the time
  subject_key_hint    bigint NULL,           -- item_key / box_key / face_key at write time: a fast path, never authoritative
  subject_entity_key  bigint NULL REFERENCES entities(entity_key) ON DELETE CASCADE,
  withdraws_claim_id  bigint NULL REFERENCES claims(claim_id),
  withdraws_op_id     uuid NULL REFERENCES label_operations(id),
  payload             jsonb NOT NULL DEFAULT '{}',   -- pydantic model per kind (section 3.3)
  CHECK ((kind = 'withdraw') = (withdraws_claim_id IS NOT NULL OR withdraws_op_id IS NOT NULL)),
  CHECK (withdraws_claim_id IS NULL OR withdraws_op_id IS NULL),
  CHECK (subject_kind NOT IN ('item','box') OR (photo_key IS NOT NULL AND subject_crop_id IS NOT NULL AND subject_bbox IS NOT NULL)),
  CHECK (subject_kind <> 'box' OR subject_box_id IS NOT NULL)
);
CREATE INDEX ix_claims_photo ON claims (project_id, photo_key, claim_id) WHERE photo_key IS NOT NULL;   -- the fold of one photo's items
CREATE INDEX ix_claims_op ON claims (op_id, claim_id) WHERE op_id IS NOT NULL;                          -- undo of an operation, label-op views (shape 46)
CREATE UNIQUE INDEX uq_claims_withdraws ON claims (withdraws_claim_id) WHERE withdraws_claim_id IS NOT NULL;  -- a claim is withdrawn once
CREATE UNIQUE INDEX uq_claims_withdraws_op ON claims (withdraws_op_id) WHERE withdraws_op_id IS NOT NULL;  -- an operation is withdrawn once
CREATE INDEX ix_claims_entity ON claims (project_id, subject_entity_key, claim_id) WHERE subject_entity_key IS NOT NULL;
CREATE INDEX ix_claims_kind_at ON claims (project_id, kind, at);           -- audit report, holdout stats, history routes
-- fillfactor 100 (never updated). Trigger claims_append_only: BEFORE UPDATE OR DELETE raises,
-- except a DELETE under op_project_delete_allows(OLD.project_id) (02a 1.2, RN-1) or the forget path (question 4).
-- 02a label_operations, changed: it groups the claims of one act
ALTER TABLE label_operations DROP COLUMN undone_at;                         -- an undo is a 'withdraw' claim naming the op
ALTER TABLE label_operations ADD COLUMN dataset_import_id uuid NULL REFERENCES dataset_imports(id) ON DELETE SET NULL;
-- status CHECK becomes ('writing','written','failed'): 'undoing'/'undone' are read from the withdraw claim
-- kind: closed list in src/domain/label_ops.py: label, batch_label, move, accept_machine, adopt_detector,
-- unlabel, discard, exclude, unexclude, dismiss, region_edit, geometry, draw, holdout_freeze, audit_start,
-- dataset_import, combine, assist_loop_propagate, duplicate_verdict, and the phase-5 gallery kinds
DROP TABLE label_operation_members;
```

Sizes at 10 M: a claim tuple is 24 + 8 + 16 + 8 + 6 + 4 + 16 + 12 (kind) + 6 + 8 + 16 + 4 + 2 x 40
(two `real[4]`: 24-byte array header + 16) + 8 + about 60 (payload) = about 290 bytes; 20 M x 294 /
8,168 = 720 k pages, **5.9 GB**. Indexes: `ix_claims_photo` 20 M x 40 B = 0.8 GB, PK 0.45 GB,
`ix_claims_op` 0.6 GB, `ix_claims_kind_at` 0.9 GB, partial ones small: **about 2.8 GB**.

### 2.3 Computed: analysers, ledger and results

```sql
CREATE TABLE analysers (                     -- deployment catalog of everything that computes: no project_id, no RLS
  analyser_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  kind text NOT NULL CONSTRAINT ck_analysers_kind CHECK (kind IN ('detector','open_vocab','region_stage','text_reader','vlm',
       'classifier','probe','score','projection','capture_parser','encoder','face_detector','assist','raw_label_clustering')),
  name text NOT NULL,                        -- detector model, VLM endpoint 'name@revision', 'capture', probe checkpoint, ...
  version text NOT NULL,                     -- detector_version, VLM model, 'capture@1', method version
  config_sha256 text NOT NULL,               -- canonical JSON of params: prompt pack revision, region profile revision, ...
  config_ref jsonb NOT NULL DEFAULT '{}',    -- {config_id, revision} of a config document, or {model_version_id}
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (kind, name, version, config_sha256)
);  -- immutable (trigger analysers_immutable): a new prompt-pack revision or model build is a new row
CREATE TABLE analysis_ledger (               -- "what has been computed for this photo": the work list of any re-run
  photo_key bigint NOT NULL REFERENCES photos(photo_key) ON DELETE CASCADE, PROJECT,
  kind text NOT NULL, analyser_id integer NOT NULL REFERENCES analysers(analyser_id),
  state text NOT NULL CONSTRAINT ck_analysis_ledger_state CHECK (state IN ('pending','done','failed','skipped')),
  reason text NULL, updated_at timestamptz NOT NULL DEFAULT now(),   -- reason: today's open_vocab_status and failures
  PRIMARY KEY (photo_key, kind)
) WITH (fillfactor = 90);
CREATE INDEX ix_analysis_ledger_work ON analysis_ledger (project_id, kind, analyser_id, state);   -- re-run work list: rows below a version
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
  CHECK ((lat IS NULL) = (lon IS NULL))
);
CREATE INDEX ix_photo_capture_taken ON photo_capture (project_id, taken_utc, photo_key);           -- shape 24 timeline sort, range filter
CREATE INDEX ix_photo_capture_geo ON photo_capture (project_id, lat, lon) WHERE lat IS NOT NULL;  -- shapes 24, 30 [gate A]
CREATE TABLE items (                         -- identity of every item; stable across analyser re-runs (section 4.1)
  item_key bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,
  photo_key bigint NOT NULL REFERENCES photos(photo_key) ON DELETE CASCADE,
  crop_id uuid NOT NULL, box_seq integer NOT NULL DEFAULT 0,   -- box_seq: today region_box_seq, never decreases
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (project_id, crop_id)
) WITH (fillfactor = 90);                    -- box_seq updates are HOT (not indexed)
CREATE INDEX ix_items_photo ON items (photo_key, item_key);
CREATE TABLE detections (                    -- one detector box; replaced when the detector re-runs
  detection_key bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,
  photo_key bigint NOT NULL REFERENCES photos(photo_key) ON DELETE CASCADE,
  item_key bigint NOT NULL REFERENCES items(item_key) ON DELETE CASCADE,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),   -- detector, or open-vocabulary set@revision
  bbox real[] NOT NULL CHECK (cardinality(bbox) = 4),
  class_name text NULL, class_id integer NULL, confidence real NULL, proposal_name text NULL,   -- detector vocabulary
  blur_lap_var real NULL, blur_lap_ratio real NULL, blur_full_var real NULL,
  prompt text NULL, mask_polygon real[] NULL,                         -- open vocabulary
  proposed_by_import_id uuid NULL REFERENCES dataset_imports(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (item_key, analyser_id)
);
CREATE INDEX ix_detections_photo ON detections (photo_key, analyser_id);
CREATE TABLE class_opinions (                -- machine class opinions: one per item and source (plan 15 section 4.3)
  item_key bigint NOT NULL REFERENCES items(item_key) ON DELETE CASCADE, PROJECT,
  source text NOT NULL CONSTRAINT ck_class_opinions_source CHECK (source IN
         ('classifier','vlm','assist_propagated','import_suggestion','adopted_detector','open_vocab_target')),
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),
  class_name text NULL, class_id_hint integer NULL,   -- NULL name: no registry class (vlm_unmatched, empty answer)
  outcome text NOT NULL CONSTRAINT ck_class_opinions_outcome CHECK (outcome IN
          ('matched','unmatched','new_class_pending','reclassified','empty','low_conf','model')),
  confidence real NULL, confidence_cat text NULL,      -- classifier score; VLM category (vlm_confidence)
  raw_label text NULL, raw_label_conf real NULL, raw_class text NULL, proposed_class text NULL,
  empty_reason text NULL, via text NULL,               -- via 'combined': class and region in one VLM call
  op_id uuid NULL REFERENCES label_operations(id) ON DELETE SET NULL,   -- propagation and adoption runs are undoable
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (item_key, source)
) WITH (fillfactor = 90);
CREATE INDEX ix_class_opinions_raw ON class_opinions (project_id, raw_label) WHERE raw_label IS NOT NULL;   -- shape 18
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
) WITH (fillfactor = 90);
CREATE INDEX ix_item_region_results_status ON item_region_results (project_id, status);   -- shapes 42, 49
CREATE TABLE region_boxes (                  -- machine boxes of the region stage
  box_key bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,
  item_key bigint NOT NULL REFERENCES items(item_key) ON DELETE CASCADE, box_id text NOT NULL,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),
  bbox real[] NOT NULL CHECK (cardinality(bbox) = 4),
  state text NOT NULL CHECK (state IN ('proposed','accepted','rejected','false_positive')),   -- BOX_STATES
  score real NULL, detector text NULL, detector_version text NULL, source text NULL,
  confidence text NULL, rejection_reason text NULL, detected_at timestamptz NOT NULL DEFAULT now(),
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
CREATE INDEX ix_text_lines_subject ON text_lines (photo_key, item_key);
CREATE INDEX ix_text_lines_trgm ON text_lines USING gin (lower(text) gin_trgm_ops);   -- text search [gate A]
CREATE TABLE probe_predictions (item_key bigint PRIMARY KEY REFERENCES items(item_key) ON DELETE CASCADE, PROJECT,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),   -- probe_model_version; params hold its class names
  class_name text NULL, class_id_hint integer NULL, entropy real NULL, confidence real NULL, margin real NULL,
  scored_at timestamptz NOT NULL DEFAULT now());
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
CREATE INDEX ix_faces_photo ON faces (photo_key);
CREATE TABLE vector_gaps (                   -- why a subject has no vector in a space (today embedding_state)
  space_id uuid NOT NULL REFERENCES embedding_spaces(id) ON DELETE CASCADE, PROJECT,
  subject_key bigint NOT NULL,
  reason text NOT NULL CHECK (reason IN ('not_selected','deferred','failed')),   -- src/domain/embedding_state.py:19-26
  PRIMARY KEY (space_id, subject_key)
);  -- a vector row's existence is "embedded"; the vector writer deletes the gap in the same transaction
```

Changes to 02d tables, in the same revision: `cluster_memberships.item_key text` becomes
`subject_key bigint NOT NULL` (an `item_key`, `box_key`, `face_key` or `photo_key` by the partition's
`target`), primary key `(partition_id, subject_key)`; no foreign key, as today (the immutability
trigger of 02d 2.2 would refuse the cascade). `near_dup_members.member_id`, `near_dup_edges.a_id`,
`b_id` and `holdout_near_dup_moves.item_id` become `bigint` keys of the run's level the same way.

Sizes at 10 M:

| Table | Tuple arithmetic | Heap | Indexes |
|---|---|---|---|
| `items` | 24 + 8 + 16 + 8 + 16 + 4 + 8 = 84, +4 → 88; 10 M x 88 / (8,168 x 0.9) | **1.2 GB** | PK 0.22, unique 0.36, photo 0.3: **0.9 GB** |
| `detections` | 24 + 8 + 16 + 8 + 8 + 4 + 40 + 8 + 4 + 4 + 8 + 12 (+ NULLs) ≈ 150; 10 M x 154 / 8,168 | **1.9 GB** | PK, unique, photo: **0.8 GB** |
| `class_opinions` | about 1 per item: 24 + 8 + 16 + 12 + 4 + 12 + 4 + 8 + 4 + 8 + 12 + 8 ≈ 120; /0.9 | **1.6 GB** | PK 0.4, raw 0.3: **0.7 GB** |
| `item_region_results` | 10 M x about 110 / (8,168 x 0.9) | **1.5 GB** | **0.5 GB** |
| `region_boxes` | 5 M x about 130 / 8,168 | **0.8 GB** | **0.3 GB** |
| `analysis_ledger` | 3 M photos x 8 kinds x 64 / (8,168 x 0.9) | **1.7 GB** | **1.0 GB** |
| `photo_capture`, `text_lines`, `probe_predictions`, `item_scores`, `item_projections`, `assist_answers`, `faces`, `vector_gaps` | filled by later or optional analysers; each below 1.5 GB at 10 M (`item_scores` 20 M x 60 B = 1.2 GB the largest) | ≤ **5 GB together** | ≤ **2 GB** |

### 2.4 Current state: the fold's output

```sql
CREATE TABLE item_state (                    -- one narrow row per item: what lists filter and sort on (plan 15 section 4.4)
  item_key bigint PRIMARY KEY REFERENCES items(item_key) ON DELETE CASCADE, PROJECT,
  photo_key bigint NOT NULL, crop_id uuid NOT NULL,           -- copies, so a list page needs no join for ids
  bbox real[] NOT NULL, area real NOT NULL, rank smallint NOT NULL,   -- effective geometry, crop_area_norm, crop_rank_in_image
  geometry_source text NOT NULL, geometry_revision integer NOT NULL DEFAULT 0,   -- 'detector' | 'human' | 'import'
  class_id integer NULL, class_source text NULL, label_source text NULL,         -- effective class; the name is joined, never copied
  class_validated boolean NOT NULL DEFAULT false, class_labeled_at timestamptz NULL,
  class_claim_id bigint NULL, class_opinion_source text NULL,  -- what set the class (provenance and label_op_id at read)
  confidence real NULL,                                         -- detector / classifier score
  detector_class_name text NULL, detector_confidence real NULL, blur_lap_ratio real NULL,   -- copies for filters
  vlm_suggestion_dismissed boolean NOT NULL DEFAULT false, needs_new_class boolean NOT NULL DEFAULT false,
  probe_disagrees boolean NULL, probe_entropy real NULL, probe_dismissed boolean NOT NULL DEFAULT false,
  excluded boolean NOT NULL DEFAULT false, excluded_reason text NULL, excluded_at timestamptz NULL,
  holdout boolean NOT NULL DEFAULT false, review_dismissed_at timestamptz NULL,
  audit_sample boolean NOT NULL DEFAULT false, audit_outcome text NULL,
  combine_conflict boolean NOT NULL DEFAULT false, import_ids uuid[] NULL,
  region_status text NULL, region_validated boolean NOT NULL DEFAULT false, region_verifier text NULL, region_label_source text NULL,
  region_count smallint NOT NULL DEFAULT 0, region_rejected_count smallint NOT NULL DEFAULT 0, region_max_score real NULL,
  region_revision integer NOT NULL DEFAULT 0,                   -- the editor's OCC token (expected_region_revision)
  cluster_partition_id uuid NULL, cluster_uid uuid NULL, cluster_state text NULL,   -- [gate A: column or side table]
  cluster_score real NULL, cluster_margin real NULL, cluster_assigned_by text NULL,
  taken_utc timestamp NULL, taken_local timestamp NULL,         -- copies of the photo's effective capture time [gate A]
  created_at timestamptz NOT NULL, updated_at timestamptz NOT NULL,
  UNIQUE (project_id, crop_id)
) WITH (fillfactor = 80);
-- Indexes, each named with the read it serves (inventory section 3); all lead with project_id:
CREATE INDEX ix_item_state_list    ON item_state (project_id, updated_at DESC, item_key) WHERE NOT excluded AND NOT holdout;  -- shape 1 default sort
CREATE INDEX ix_item_state_class   ON item_state (project_id, class_id, class_validated);                                 -- shapes 1, 16, 47; class counts
CREATE INDEX ix_item_state_photo   ON item_state (photo_key, rank);                                                       -- shape 25 card facts, shape 29 photo detail
CREATE INDEX ix_item_state_review  ON item_state (project_id, class_source, class_validated) WHERE NOT excluded AND review_dismissed_at IS NULL;  -- shape 2 tabs
CREATE INDEX ix_item_state_cluster ON item_state (project_id, cluster_uid, cluster_score DESC) WHERE cluster_uid IS NOT NULL;  -- shapes 10, 11, 12, 13 [gate A]
CREATE INDEX ix_item_state_region  ON item_state (project_id, region_status);                                               -- shapes 6, 15, 42
CREATE INDEX ix_item_state_taken   ON item_state (project_id, taken_utc, item_key) WHERE taken_utc IS NOT NULL;              -- shape 1 capture range [gate A]
CREATE INDEX ix_item_state_imports ON item_state USING gin (import_ids) WHERE import_ids IS NOT NULL;                      -- shape 1 import filter
CREATE TABLE box_state (                     -- one row per live box: machine box folded with box claims
  box_row_key bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,
  item_key bigint NOT NULL REFERENCES items(item_key) ON DELETE CASCADE,
  box_id text NOT NULL,
  box_key bigint NULL REFERENCES region_boxes(box_key) ON DELETE SET NULL,   -- NULL: a box that exists only through a claim
  bbox real[] NOT NULL, state text NOT NULL, score real NULL, detector text NULL, detector_version text NULL,
  source text NULL, bbox_correct boolean NULL, confidence text NULL, rejection_reason text NULL,
  text text NULL, text_source text NULL, text_choice text NULL, text_disagreement boolean NULL,
  text_fold text NULL,                       -- today's op_text_compact normalisation, written by one domain function
  cluster_uid uuid NULL, cluster_state text NULL, cluster_score real NULL,   -- [gate A, as item_state]
  ordinal smallint NOT NULL, detected_at timestamptz NULL,
  UNIQUE (item_key, box_id)
) WITH (fillfactor = 85);
CREATE INDEX ix_box_state_state ON box_state (project_id, state, detector);                 -- shapes 6, 15, 52
CREATE INDEX ix_box_state_text_trgm ON box_state USING gin (text_fold gin_trgm_ops) WHERE text_fold IS NOT NULL;   -- shape 2 wildcard [gate A]
```

No `INCLUDE` payload columns: the table is rewritten on every label, and "there is little point in
including payload columns in an index unless the table changes slowly enough"
([index-only scans](https://www.postgresql.org/docs/18/indexes-index-only-scans.html)); slow count
reads get summary tables (section 8). Trigram indexes use `pg_trgm`
([pg_trgm](https://www.postgresql.org/docs/18/pgtrgm.html)). Both tables are derived (R7):
`fold.rebuild(project)` deletes and refolds them, and a test compares answers; `items` is not emptied
(section 4.1). Fillfactor 80: changes to non-indexed columns (probe entropy, region counts) stay HOT;
a label changes the indexed `class_id` and is never HOT, but the free space keeps the new version on
the same page more often.

Sizes at 10 M: `item_state` tuple ≈ 24 header + 16 project + 3 x 8 keys + 16 crop id + 40 bbox + 30
fixed numbers + 10 booleans + 48 (six timestamps) + 32 (two cluster uuids) + about 90 of short text ≈
330 bytes; 10 M x 334 / (8,168 x 0.8) = 511 k pages, **4.2 GB**; eight B-trees of 25-45 B per entry
at 90 % fill, 0.3-0.5 GB each, **about 3.2 GB**. `box_state`: 5 M x 150 / (8,168 x 0.85) = **1.1 GB**,
indexes **0.5 GB**.

### 2.5 Gallery: entities, edges, photo state (phase 5; designed now, created now, empty until then)

```sql
CREATE TABLE entities (                      -- people, places, events, trips, albums, keywords
  entity_key  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,
  entity_id   uuid NOT NULL DEFAULT uuidv7(),   -- wire id (persons.id, album_id of photo_library 00 section 6.1)
  kind        text NOT NULL CONSTRAINT ck_entities_kind CHECK (kind IN ('person','place','event','trip','album','keyword')),
  analyser_id integer NULL REFERENCES analysers(analyser_id),   -- set when a machine made it (grouping, events, places)
  cluster_uid uuid NULL REFERENCES cluster_identities(cluster_uid) ON DELETE SET NULL,  -- a person's face cluster identity
  geonames_id integer NULL,                   -- a place: geo_places row (photo_library 10)
  created_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (project_id, entity_id)
);  -- A person created by a person (split) or an album is created by an 'entity_created' claim.
CREATE TABLE entity_edges (                  -- machine links of a subject to an entity: results
  entity_key  bigint NOT NULL REFERENCES entities(entity_key) ON DELETE CASCADE, PROJECT,
  subject_kind text NOT NULL CHECK (subject_kind IN ('photo','face','item')),
  subject_key bigint NOT NULL,
  analyser_id integer NOT NULL REFERENCES analysers(analyser_id),
  weight      real NULL,
  PRIMARY KEY (entity_key, subject_kind, subject_key)
);
CREATE INDEX ix_entity_edges_subject ON entity_edges (project_id, subject_kind, subject_key);   -- both directions
CREATE TABLE entity_state (                  -- fold of entity claims: name, merged into, hidden
  entity_key bigint PRIMARY KEY REFERENCES entities(entity_key) ON DELETE CASCADE, PROJECT,
  name text NULL, hidden boolean NOT NULL DEFAULT false, merged_into bigint NULL, state text NOT NULL,
  member_count integer NOT NULL DEFAULT 0, updated_at timestamptz NOT NULL
) WITH (fillfactor = 80);
CREATE TABLE membership_state (              -- fold of entity_edges and membership claims (albums, keywords, person confirm/reject)
  entity_key bigint NOT NULL REFERENCES entities(entity_key) ON DELETE CASCADE, PROJECT,
  subject_kind text NOT NULL, subject_key bigint NOT NULL,
  source text NOT NULL CHECK (source IN ('machine','human')), position numeric NULL, weight real NULL,
  PRIMARY KEY (entity_key, subject_kind, subject_key)
);
CREATE INDEX ix_membership_state_subject ON membership_state (project_id, subject_kind, subject_key);
CREATE TABLE face_state (                    -- fold: the face's person (grouping result + confirm/reject claims)
  face_key bigint PRIMARY KEY REFERENCES faces(face_key) ON DELETE CASCADE, PROJECT,
  person_entity_key bigint NULL REFERENCES entities(entity_key) ON DELETE SET NULL,
  person_state text NOT NULL CHECK (person_state IN ('unassigned','suggested','assigned','confirmed')),
  assign_score real NULL, assign_source text NULL
) WITH (fillfactor = 80);
CREATE TABLE photo_state (                   -- fold of photo claims over photo_capture: what the timeline and map read
  photo_key bigint PRIMARY KEY REFERENCES photos(photo_key) ON DELETE CASCADE, PROJECT,
  taken_local timestamp NULL, taken_utc timestamp NULL, time_source text NULL,   -- effective: correction claim, else read
  lat double precision NULL, lon double precision NULL, location_source text NULL,
  favourite boolean NOT NULL DEFAULT false, pick text NOT NULL DEFAULT 'none', hidden boolean NOT NULL DEFAULT false,
  archived boolean NOT NULL DEFAULT false, trashed_at timestamptz NULL,
  title text NULL, caption text NULL, updated_at timestamptz NOT NULL
) WITH (fillfactor = 80);
CREATE INDEX ix_photo_state_timeline ON photo_state (project_id, taken_utc DESC, photo_key) WHERE trashed_at IS NULL AND NOT hidden;
```

These replace, for the gallery, the mutable `photo_annotations`, `capture_overrides`, `media_text`,
`album_members`, `persons`, `face_constraints` and `library_operation_members` of
[photo_library 00](../photo_library/00_photo_library_plan.md) section 6.1, because plan 15 R3 makes a
human decision an appended row (question 3). Smart albums (a revisioned rule document), media tracks
and points (added data) and `geo_places` (deployment data) stay as photo_library 00 designs them.
Sizes at 10 M: empty in phase 2; at 3 M photos `photo_state` ≈ 3 M x 120 / (8,168 x 0.8) = 0.55 GB.

## 3. The claim log

### 3.1 Columns

DDL in section 2.2. A claim names: when (`at`, and `claim_id` for order), who (`actor_kind`,
`actor`), the operation it belongs to (`op_id`), what (`kind`, `payload`) and the subject.

### 3.2 Kinds

One closed list in `src/domain/claim_kinds.py`, equal to `ck_claims_kind` by test.

| Kind | Subject | Today's route or writer (decorator file:line) |
|---|---|---|
| `item_drawn` | item | `POST /images/{image_id}/items` (`src/routers/curation/image_items.py:21`); an import creating an item (`dataset_import/item_labels.py:179`) |
| `geometry_set` | item | `PATCH /crops/{crop_id}/geometry` (`crops_geometry.py:39`) |
| `class_set` | item | `PUT /crops/{crop_id}/label` (`crops_write.py:39`), `PUT /crops/batch_label` (`:108`), `POST /label_ops/accept_machine` (`label_ops.py:151`), `POST /review/new_class_proposals/resolve` (`review_resolve.py:111`), validated import labels |
| `class_cleared` | item | `DELETE /crops/{crop_id}/label` with no human label on record, `POST /crops/{crop_id}/discard` with `clear_class` (`label_undo.py:109,:131`) |
| `class_moved` | item | `POST /crops/move` (`crops_write.py:161`) |
| `excluded`, `restored` | item | `POST /crops/batch_exclude`, `/crops/batch_unexclude` (`crops_exclude.py:29,:68`) |
| `review_dismissed`, `review_restored` | item | `/review_dismiss` (`crops_exclude.py:95`), `/discard` with `dismiss_from_review`, `/review_undismiss` (`label_undo.py:228`) |
| `vlm_dismissed`, `probe_dismissed` | item | `/vlm_dismiss` (`label_undo.py:180`), `/probe_dismiss` (`:197`) |
| `needs_new_class` | item | `POST /crops/flag_new_class` (`crops_write.py:216`) |
| `holdout_frozen` | item | `POST /test_holdout/freeze` (`holdout_freeze.py:19`); import and combine freezes |
| `audit_sampled` | item | `POST /audit/start` (`audit.py:23`) |
| `region_status_set` | item | `PATCH /crops/{crop_id}/region_meta` (`regions_edit.py:118`), `POST /regions/batch_status` (`:272`) |
| `box_created`, `box_moved`, `box_state_set`, `box_text_set`, `box_deleted` | box | `PUT /crops/{crop_id}/regions` (`regions_boxes_edit.py:291`, as a diff), `PUT /crops/batch_regions` (`:335`), `PATCH /crops/{crop_id}/regions/{box_id}` (`:368`), `POST /regions/batch_box_state` (`:426`); validated import boxes |
| `combine_conflict` | item | the combine job (`projects/combine/*`) |
| `duplicate_verdict` | pair | `POST /near_duplicates/decisions` (`near_duplicates.py:110`); the `DELETE` is a `withdraw` |
| `withdraw` | claim or operation | every undo route: `/label/undo`, `/label/undo_batch`, `/label_ops/{op_id}/undo`, region, geometry, dismiss undos (`label_undo.py:62,:78`, `label_ops.py:49`, `crops_geometry.py:64`, `edit_undo.py:156,:175`), import undo, `forget_holdout_freeze` |
| phase 5: `favourite_set`, `pick_set`, `hidden_set`, `archived_set`, `trashed`, `untrashed`, `capture_corrected`, `text_set`, `entity_created`, `entity_named`, `entity_merged`, `entity_hidden`, `member_added`, `member_removed`, `face_confirmed`, `face_rejected` | photo, entity, face | the library routes of photo_library 00 section 7; the person routes `PUT /persons/{person_id}/name`, `POST /persons/merge` (`src/routers/persons.py`) for the core people |

Class registry edits (create, rename, deprecate, merge, hotkeys) stay rows of `project_classes` with
their revision (02a section 3): they are configuration, not a decision about an object.

### 3.3 Payloads (pydantic models in `src/domain/claim_payloads.py`, `extra='forbid'`)

| Kind | Payload | Note |
|---|---|---|
| `item_drawn` | `{bbox, request_id?, standalone_region?: bool, origin?: {project, crop_id, image_id}}` | `subject_crop_id` = `drawn_crop_id(image_id, request_id)`; an import's item uses the detector-style `crop_id` |
| `geometry_set` | `{bbox}` | `subject_machine_bbox` = the detector box at the time (today `detector_bbox_norm`) |
| `class_set` | `{class_name, class_id_hint, label_source, via: 'single'\|'batch'\|'bulk_accept'\|'resolve'\|'import'\|'combine'}` | `label_source` is one of `HUMAN_LABEL_SOURCES`, `human_bulk_accept` or `import` |
| `class_cleared`, `excluded`, `restored`, `review_*`, `needs_new_class` | `{}`, `{reason}`, `{}`, `{}`, `{note?}` | `excluded.reason` `not_an_object` is a delete |
| `class_moved` | `{partition_id, cluster_uid}` | |
| `vlm_dismissed` | `{class_name?, class_id_hint?}` | the suggestion that was rejected |
| `probe_dismissed` | `{class_name, probe_analyser_id}` | |
| `holdout_frozen` | `{freeze_id, machine_class: {class_name, source, confidence}?}` | the machine class that would win at the freeze (section 4.3) |
| `audit_sampled` | `{batch_id, label_name, label_source}` | |
| `region_status_set` | `{status: 'confirm'\|'reject'\|'false_positive', label_source}` | |
| `box_created` | `{bbox, state, text?}` | `subject_box_id` minted from `items.box_seq` |
| `box_moved`, `box_state_set`, `box_text_set` | `{bbox, bbox_correct?}`, `{state, rejection_reason?}`, `{text}` | an explicit empty text is a reading of "no text" (`region_boxes.py:277`) |
| `combine_conflict` | `{origins: [..], merged_origins: [..]}` | |
| `duplicate_verdict` | `{level, a, b, decision: 'same'\|'different'}` | `a`, `b` are wire ids |
| `withdraw` | `{}` | the target is `withdraws_claim_id` or `withdraws_op_id` |

Class in a claim: by name, with the registry id of that moment as a hint. Resolution at fold time:
the hint if `project_classes(project, hint)` exists (following `merged_into` to the end), else the
active class whose `name_key = normalize_class_name(name)` (02a section 3 rule), else the claim
**creates nothing** and the fold reports the item `class_unresolved` (fail closed; the claim stays).

### 3.4 Addressing a subject after every result was rebuilt

Photos are never rebuilt, so `photo_key` is as stable as the wire `image_id`. An item or box is
found again by this rule (the one function `src/domain/claim_match.py`, reused by the fold, by
re-detect and by import, replacing the matching of `src/services/curation/proposal_merge.py:109-170`):

1. An item of the photo whose `crop_id = subject_crop_id` exists: that item.
2. Else greedy matching over all unmatched claims and new detections of the photo, best IoU first,
   ties by `crop_id` then claim id: IoU of the detection box with `subject_bbox`, or with
   `subject_machine_bbox` when the claim has one, at least **0.5** (`LABEL_IOU_MATCH`,
   `proposal_merge.py:30`, the threshold re-detect and import use today). The matched detection's
   item takes the claim's `subject_crop_id` as its `crop_id` when it is new, so an item a person
   touched keeps its wire id across detector versions (today a locked item is never replaced,
   `reprocess_detect.py:1-11`).
3. No match: a claim that asserts an object (`item_drawn`, `geometry_set`, `class_set`,
   `holdout_frozen`, any box claim) creates an item from the claim alone, with `geometry_source`
   `human` or `import` and the claim's geometry; nothing a person decided is lost. A claim that
   only removes or hides (`excluded` with `not_an_object`, `review_dismissed`, `class_cleared`)
   stays dormant and applies to the first later detection that matches it, so a deleted object
   stays deleted when the detector finds it again (today the excluded document blocks re-creation,
   `occ_locks.py:90-99`).
4. Boxes: the same rule inside the item, on `subject_box_id` then box IoU ≥ 0.5.

### 3.5 Undo, operations, imports, retention

- **Undo** is a `withdraw` row naming one claim (`withdraws_claim_id`) or a whole operation
  (`withdraws_op_id`: one row withdraws 100,000 claims). The fold ignores withdrawn claims; a later
  claim on the same item still wins, so an undo can never revert a newer decision (today the undo
  guards with fingerprints, `label_ops/views.py:34`). The labeler's step-back undo is "withdraw my
  newest live claim of this kind on this item": the stack walk of `find_undo_snapshot`
  (`src/domain/history.py:219`) becomes one indexed read. A redo appends a copy of the claim. A
  `withdraw` is never withdrawn.
- **Operations**: every human act writes one `label_operations` row (kind as in section 2.2) and its
  claims carry `op_id`. Single labels too (today single labels are recorded ops, kept 1,000,
  `label_ops/store.py:55-58`); nothing is pruned any more.
- **Import**: a validated import label is a `class_set` with `actor_kind = 'import'`,
  `actor = 'import:<id>'`, `op_id` of the import's operation (which names `dataset_import_id`). An
  unvalidated (`label_trust: suggestion`) label is a `class_opinions` row, source
  `import_suggestion` (plan 15 section 4.2). Import undo is one `withdraw` of the operation: items
  that only the import's claims created disappear, items a person also claimed stay
  (`is_human_owned_item`, `occ_locks.py:154`, becomes "has a live human claim").
- **Retention**: none (R3). Volume: a labelling session of one hour labels about 2,000 items (cluster
  by cluster, bulk) and writes about 2,100 claims (labels, a few undos, excludes) and 5-20 operation
  rows; a holdout freeze of 10 % writes one claim per frozen item (1 M at 10 M). Twenty million
  claims over a 10 M project's life is the sizing figure.

## 4. The fold

### 4.1 Inputs and outputs

`fold_item(inputs) -> ItemState | None` in `src/domain/fold.py`, pure (no I/O, no clock), with
`fold_boxes` and `fold_photo` beside it. Inputs for one item: its `items` row; its detections; its
class opinions; its region result and machine boxes with readings; its probe prediction; its
cluster placement in the serving partition; the photo's capture copy; the live claims attached to it
(after `claim_match` and the withdraw filter); the class registry snapshot. Output: the `item_state`
row and the item's `box_state` rows, or `None` when the item has neither a current detection nor a
live claim (its `items` row is then deleted in the same transaction).

The step before the fold, `attach(photo)`, runs `claim_match` and inserts or deletes `items` rows.
It is the only writer of `items`. `fold.rebuild` deletes `item_state` and `box_state` and folds
every item; it never deletes an `items` row that still has an input.

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

Inside one source the newest `analyser_id` row wins (one row per source by key). One order is
undefined today (whichever of propagation and an import suggestion wrote last): rank 3 over 4 here,
because an import label is about this item and a propagated one is carried over from its cluster
(question 5).

### 4.3 The fold, layer by layer

1. **Existence and geometry**: a live `item_drawn` or `geometry_set` (newest wins) gives the bbox and
   `geometry_source` (`human`, or `import` for an import actor); else the attached detection.
   `area` and `rank` (rank among the photo's live items by area, `item_geometry.py:219`) follow.
2. **Class**: the newest live claim among `class_set`, `class_cleared`, `class_moved` decides when
   one exists: `class_set` → that class, `class_source` `human` (`external_label` for an import
   actor), validated; `class_cleared` → no class, no machine class shown (the person said so;
   today's reset, `label_undo.py:109-121`); `class_moved` → no claimed class, the machine class of
   4.2 stays as a suggestion, validated false, placement from the claim (`candidate_move_update`,
   `class_label.py:239-277`). Else, when a live `holdout_frozen` exists, its `machine_class`. Else
   `machine_class` of 4.2.
3. **Flags**: `excluded` and `review_dismissed_at` from the newest live `excluded`/`restored` and
   `review_dismissed`/`review_restored`; `holdout` = a live `holdout_frozen`; `needs_new_class` = a
   live flag claim, or opinion outcome `new_class_pending`, until a later `class_set`;
   `vlm_suggestion_dismissed` = a live `vlm_dismissed` naming the current VLM suggestion
   (`vlm_suggestion_dismissed`, `class_sources.py:198`); `probe_dismissed` likewise against the
   current probe class; `audit_sample` and `audit_outcome` from `audit_sampled` and the first
   `class_set` after it (`audit_outcome_fields`, `src/domain/audit_math.py`); `combine_conflict` = a
   live conflict claim with no later `class_set`; `import_ids` = the imports of the live claims.
4. **Boxes**: machine boxes matched to box claims; per box the newest live claim of each aspect wins
   (geometry, state, text); `source = 'human'` or `'import'` for a claim-created box;
   `rejection_reason = REJECT_REASON_HUMAN` for a human reject; `text_source = 'human'` for a typed
   text; else the reader consensus of `box_texts` (the existing choice function of
   `src/services/detection/region_text.py:400-455`, moved to `src/domain/`). Item summary:
   `region_status` = a live `region_status_set` if any, else `derive_status` over the folded boxes
   (`region_boxes.py:307`), else the machine status; counts and max score as `boxes_write_fields`
   (`region_boxes.py:329-382`); `region_validated` = a live `region_status_set` or any live human box
   claim; `region_verifier` = `human`/`import` from that claim's actor, else the machine verifier.
5. **Placement**: a live `class_moved` → its cluster with `cluster_assigned_by = 'edit'`; else the
   membership of the serving partition.
6. **Revisions**: `geometry_revision` = the number of geometry claims ever written for the item plus
   the withdraws of them; `region_revision` = the same count over box and region claims plus the
   number of region results applied. Counting withdrawn claims and the withdraws keeps both strictly
   increasing, as today (a restore "moves forward from the stored revision", `edit_history.py:79-80`):
   the editor's `expected_region_revision` check and the thumbnail URL's `?v=` (`wire.py:470`) rely
   on it. `updated_at` = the newest `at`/`created_at` among the inputs.

### 4.4 Each lock predicate is implied by the data shape

Machine writers insert or replace only result rows (sections 2.3); they never write `claims`, and
the fold puts every claim layer above every result layer. So for each predicate of
`src/clients/occ_locks.py`, the state it protects comes from a claim, which no machine write can
change:

| Predicate (`occ_locks.py`) | Today true when | In the record it is true when | Why no machine write can change the protected state |
|---|---|---|---|
| `is_human_marker` (:22) | a guard field contains `human` | `class_source` is `human` because layer 2 found a live human `class_set` | layer 2 reads claims before opinions; opinions cannot remove a claim |
| `is_locked_class` (:44), human | `class_source` human | as above | as above |
| `is_locked_class`, import | `external_label` and validated | live `class_set` with `actor_kind = 'import'` | as above; an import suggestion is an opinion and stays replaceable, as today |
| `is_locked_class`, holdout | `test_holdout` | live `holdout_frozen` | layer 2 uses the frozen `machine_class` before any current opinion |
| `is_locked_box` (:77) | box `source` human, human reject, human text, or import box not proposed | the box aspect comes from a live box claim | layer 4 reads box claims before `region_boxes` and `box_texts` |
| `is_human_deleted` (:90) | excluded, by human, `not_an_object` | live human `excluded` with that reason | layer 3; a dormant claim re-applies to a new matching detection (3.4) |
| `has_human_geometry` (:102) | `geometry_source` human or deleted | live human `item_drawn`/`geometry_set`, or deleted | layer 1 reads claims before detections |
| `has_import_geometry` (:109) | `import_ids` and validated `external_label` | live import `item_drawn` or `class_set` | layer 1 and 2 |
| `is_locked_geometry` (:121) | the two above or holdout | the claims above or live `holdout_frozen` | holdout freezes the item's existence: `claim_match` step 3 keeps it |
| `is_locked_item` (:137) | any of the above, or region validated by human/import verifier | any of the above, or a live `region_status_set` / human box claim | layer 4 |
| `is_human_owned_item` (:154) | human class, human geometry, curator holdout, human box, human-validated set | the item has a live claim with `actor_kind = 'human'` | used by import undo: the item survives the withdrawal |

The wire flags `label_locked` and per-box `locked` (`wire.py:389`, `:525`) keep today's predicates,
evaluated on the folded state; one difference is intended: a human accept of a machine box is now a
claim, so no machine writer can touch it, while the wire `locked` stays false for it as today
(`region_boxes.py:283-305`).

### 4.5 Property tests the lane writes (Hypothesis, `tests/domain/test_fold_properties.py`)

1. Any permutation of machine writes (opinions, detections, region results, probe) interleaved with
   the same claims gives the same `item_state` and `box_state`.
2. A withdrawn claim leaves no trace: fold(claims + c + withdraw(c)) = fold(claims).
3. A withdrawn operation equals withdrawing each of its claims.
4. On every generated history, each predicate of the table in 4.4, evaluated on the folded state,
   is unchanged by any further machine write.
5. Emptying every result and re-running produces the same state for every item that has a live
   claim (with the matching rule of 3.4 and detections jittered under IoU 0.5).
6. `rebuild` equals incremental folding after every write.
7. Class resolution follows `merged_into` and survives rename; an unresolvable name yields
   `class_unresolved`, never another class.
8. `region_revision` strictly increases on every write the editor can observe.
9. Ranking: removing the top-ranked opinion yields the next one (4.2), never a lower rank.

## 5. Field map

Classes: M machine, H human, I identity or provenance, D derived. "read" = derived at read from the
named source. Today's fields from inventory 1.1 and 1.2, checked against
`src/clients/curation_opensearch/bodies_core.py`, `items_extra.py`, `capture_fields.py`,
`assist_fields.py`, `open_vocab_fields.py`, `src/clients/opensearch/indexes.py`.

### 5.1 `images` document (62 fields)

| Today's field | Class | Destination |
|---|---|---|
| image_id, imohash, width, height, decode_route, source, source_identifier | I/M | `photos.` same names (`imohash` as bytes) |
| image_path | I | `photos.storage_key` (R9) |
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
| counts.items, labeled, validated, unlabeled, in_holdout | D | read: aggregate of `item_state` by `photo_key` (a summary table only if gate A asks) |
| counts_at | D | deleted: bookkeeping of the Valkey dirty set, which goes |

### 5.2 `items` document (top level)

| Today's field | Class | Destination |
|---|---|---|
| crop_id; image_id; created_at; region_box_seq | I/D | `items.crop_id` (copy in `item_state`); `items.photo_key` → `photos.image_id`; `items.created_at`; `items.box_seq` |
| request_id | I | `photos.ingest_request_id`; a drawn item's in `claims.payload.request_id` |
| image_path, source | I | read: `photos.storage_key`, `photos.source` |
| bbox_norm, crop_area_norm, crop_rank_in_image, geometry_source, geometry_revision | M/H/D | `item_state.bbox, area, rank, geometry_source, geometry_revision` (fold layer 1) |
| detector_bbox_norm | M | `detections.bbox`; at claim time `claims.subject_machine_bbox` |
| class_id, class_source, label_source, class_validated, class_labeled_at, confidence | H/M | `item_state.` same names (fold layer 2) |
| updated_at, test_holdout, needs_new_class, review_dismissed_at, import_ids, combine_conflict, audit_sample, audit_outcome | D/H | `item_state.updated_at, holdout, needs_new_class, review_dismissed_at, import_ids, combine_conflict, audit_sample, audit_outcome` |
| class_excluded, excluded_at, excluded_reason | H | `item_state.excluded, excluded_at, excluded_reason` |
| region_status, region_validated, region_label_source, region_verifier, region_count, region_rejected_count, region_max_score, region_revision | D/H | `item_state.` same names (`region_verifier`: human/import from a claim, else the machine's) |
| cluster_partition_id, cluster_uid, cluster_state, cluster_score, cluster_margin, cluster_assigned_by | D | `item_state.` same names [gate A: or read from `cluster_memberships` of the serving partition] |
| capture_taken_local, capture_taken_utc | D | `item_state.taken_local, taken_utc` (copies of the photo's effective capture time) |
| probe_disagreement; probe_pred_entropy | D/M | `item_state.probe_disagrees` (fold: prediction against the effective class, null outside the probe's class names); `probe_predictions.entropy` (copy `item_state.probe_entropy`) |
| class_name | H/M | read: `project_classes.name` by `item_state.class_id` |
| class_detector, class_detector_version, class_labeler | I | read: the winning claim's actor (human detector name, `class_label.py:45`) or the winning opinion's analyser |
| label_op_id | I | read: `claims.op_id` of `item_state.class_claim_id` (or the opinion's `op_id`) |
| detector_class_name, detector_class_id, detector_confidence, proposal_name, blur_lap_var, blur_lap_ratio, blur_full_var, source_prompt, mask_polygon | M | `detections.class_name, class_id, confidence, proposal_name, blur_lap_var, blur_lap_ratio, blur_full_var, prompt, mask_polygon` (filter copies in `item_state`) |
| proposed_by_import | I | `detections.proposed_by_import_id` |
| open_vocab_set, open_vocab_revision | I | read: the open-vocabulary detection's `analysers` row |
| vlm_confidence, vlm_raw_label, vlm_raw_label_conf, vlm_raw_class, vlm_proposed_class, vlm_class_attempted_at, vlm_class_empty_reason, vlm_verify_completed_at | M/I | `class_opinions.confidence_cat, raw_label, raw_label_conf, raw_class, proposed_class, created_at, empty_reason`, `created_at` where `via = 'combined'` |
| vlm_endpoint, vlm_model, vlm_prompt_pack | I | read: the VLM opinion's `analysers` row (name, version, config) |
| vlm_label_cluster_id, vlm_label_cluster_name, vlm_label_cluster_distance | M | `raw_label_clusters.cluster_id, cluster_name, distance`, joined on `raw_label` |
| vlm_dismissed_class_id, vlm_dismissed_class_name, vlm_dismissed_at | H | `claims` (`vlm_dismissed` payload, `at`); flag `item_state.vlm_suggestion_dismissed` |
| needs_new_class_note, needs_new_class_at (written but unmapped, `crops_write.py:237-239`) | H | `claims.payload.note`, `claims.at` |
| probe_dismissed_at, probe_dismissed_class, probe_dismissed_model_version | H | `claims` (`probe_dismissed`); flag `item_state.probe_dismissed` |
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
| dataset_split, import_source_stem, import_stratum, import_hard_negative, imported_at, import_dataset_name, import_dataset_sha, on_negative_frame, proposal_chain | I/D | read: `photo_provenance` (split, stem, stratum, hard negative, `label_state = 'negative'`); earliest live import claim `at`; `dataset_imports.dataset_name`; `dataset_uploads.sha256`; analysers of the detections attached to a claimed item |
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
| excluded_prior_class_validated | H | deleted: restore is a withdraw; the fold recomputes validation |

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
| cluster_partition_id, cluster_uid, cluster_state, cluster_score, cluster_margin, cluster_assigned_by | D | `box_state.cluster_uid, cluster_state, cluster_score` [gate A]; partition, margin, assigned_by read from `cluster_memberships` |

### 5.4 Global indexes (1.2)

| Index: field | Class | Destination |
|---|---|---|
| global: image_id, imohash, width, height | I/M | `photos.` (the core project, question 1) |
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
| `items` top level (5.2; the 2 unmapped `needs_new_class_*`, 10 `assist.*` leaves, 3 `region_box_embeddings` leaves and the `vec_{key}` pair counted one each) | 175 | 132 | 33 | 10 (`class_id_history`, `label_validated`, `region_status_before_requeue`, `region_class_id`, `region_pairing`, `region_run_id`, `region_run_verify`, `item_text_tokens`, `excluded_prior_class_validated`, `edit_history`) |
| `region_boxes[]` (5.3) | 32 | 28 | 3 | 1 (`text_norm`) |
| global indexes (5.4: 17 + 13 + 12 + 22 + 12) | 76 | 50 | 24 | 2 (`is_reference`, `thumbnail_b64`) |
| **total** | **345** | **252** | **72** | **21** |

Every "no writer found" field of inventory section 9 was re-checked with `rg` on this tree:
`embedding` and `non_jpeg_format_skipped` (images), `label_validated` (items), and additionally
`region_class_id`, `region_pairing`, `is_reference`: deleted. `region_visible` has a writer
(`scripts/curation/worker/verify.py:207`) and is kept. The `OP_REGION_FIELD_*` storage-name override
of `RegionFields.from_env` (`src/config/region_fields.py:117`) has no meaning for fixed columns; the
lane deletes it and lists the variables in `src/config/retired_env.py`.

## 6. Vectors

### 6.1 Template (one table per embedding space)

```sql
-- created by op_vector_space_create(space_id) (question 2); name 'v<projects.lock_key>_<space_key>' (≤ 60 chars)
CREATE TABLE v<lock_key>_<space_key> (
  subject_key bigint PRIMARY KEY, PROJECT,   -- item_key, box_key, photo_key or face_key by embedding_spaces.target
  input_rev   integer NOT NULL DEFAULT 0,   -- geometry_revision (item) or box claim count (box) the vector was computed from
  embedding   halfvec(<dim>) NOT NULL
);
ALTER TABLE v<lock_key>_<space_key> ALTER COLUMN embedding SET STORAGE PLAIN;
-- the creating function also enables and forces RLS(02a) and grants op_app, as for every project table
-- fillfactor 100: a row is inserted, replaced whole (DELETE + INSERT on a new build), never updated in place
-- The approximate index is gate B's (section 8): e.g. CREATE INDEX ... USING hnsw (embedding halfvec_cosine_ops)
```

No foreign key to the subject: a subject deleted by the attach step (section 4.1) deletes its
vectors in the same transaction through the repository (one `DELETE ... WHERE subject_key = ANY`),
because the template cannot know which table its key points to. A check job
(`fold.verify_orphans`) counts vectors without a subject. The model version is the space's
(`embedding_spaces.model_version_id`): a space is one model version by definition
(`src/domain/embedding_spaces.py:1-6`), so R6's "version beside it" is the space row, not a column
repeated 10 million times.

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

| Transition | Storage action, in the same transaction as the status change |
|---|---|
| (insert) `pending` | `embedding_spaces` row; `op_vector_space_create(space_id)` creates the table from the template |
| `pending` → `building` | the reindex job (`embedding_reindex_runs`) fills it in batches of 2,000 through the staging pattern of section 1, deleting `vector_gaps` rows it fills |
| `building` → `ready` | gate B's index build (section 8), then `role` may become `primary` (`uq_embedding_spaces_one_primary`) |
| `ready` → `retiring` → `retired` | readers stop at `retiring`; at `retired`, `op_vector_space_drop(space_id)` drops the table (one `DROP TABLE`, no row-by-row delete) and the space's `vector_gaps` rows go by cascade |
| `failed` → `building` | the table is truncated by the job and filled again |

`embedding_spaces` change: drop `index_role`, `vector_field`, `state_field`, `nested_path`,
`knn_method` and their two `CHECK`s; add `vector_table text NOT NULL UNIQUE` and
`vector_index jsonb NOT NULL` (a `VectorIndexSpec` pydantic body: method, operator class,
parameters; gate B owns its values). The `KnnMethodSpec` model and `physical_names` /
`legacy_names` (`embedding_spaces.py:70-135`) are deleted in the vector lane.

### 6.4 Reading every vector of a scope

```sql
COPY (
  SELECT v.subject_key, v.embedding
  FROM   v<lock_key>_<space_key> v
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

### 6.5 Sizes at 10 M

| Space | Rows | Heap (PLAIN) | Primary key | Approximate index (gate B, estimate) |
|---|---|---|---|---|
| item, 1,024 dims | 10 M | 27.3 GB | 10 M x 20 B / 0.9 ≈ 0.22 GB | HNSW on `halfvec`, m 16: about 2,056 + 2 x 16 x 6 B per element at about 3 per page ≈ 27 GB, built in `maintenance_work_mem` of that order ("hnsw graph no longer fits into maintenance_work_mem" slows the build, [pgvector](https://github.com/pgvector/pgvector); default 64 MB, [resource settings](https://www.postgresql.org/docs/18/runtime-config-resource.html)); on 1-bit codes `binary_quantize(embedding)::bit(1024)` with float16 rescore: 128 + about 200 B per element ≈ 3.5 GB |
| frame, 1,024 dims | 3 M | 8.2 GB | 0.07 GB | as above, x 0.3 |
| box, 1,024 dims | 5 M | 13.7 GB | 0.11 GB | as above, x 0.5 |

## 7. Transactions

All write paths run under `project_tx` (02a 1.4) at READ COMMITTED. Nothing assumes one process:
the API runs about 30 worker processes, plus worker containers. **Lock order, everywhere:** (1) the
`label_operations` row being written, (2) `class_registries` row (only class creation and merge),
(3) `photos` rows in ascending `photo_key`, (4) `items` rows in ascending `item_key`
(`SELECT ... FROM items WHERE item_key = ANY(:keys) ORDER BY item_key FOR UPDATE`), then only
inserts and updates of rows those locks cover. Every writer takes locks in this one order, so two
writers can wait but never deadlock (the documented defence: "acquire locks on multiple objects in a
consistent order", [explicit locking, deadlocks](https://www.postgresql.org/docs/18/explicit-locking.html)). The `items` row lock
is what keeps the fold correct: a fold reads its inputs only after it holds the lock, so a second
writer waits, then reads the first one's committed inputs (READ COMMITTED: each statement sees rows
committed before it starts, 02a 1.3) and folds them in. Without it, two concurrent folds could each
miss the other's input and the later `UPDATE` of `item_state` would lose a claim's effect.

| Act | Statements, in order | Locks | A concurrent second writer |
|---|---|---|---|
| **Ingest of a commit group** (13a L4) | class creation before the transaction (barrier B3 of 13a, `ensure_class` locks `class_registries`); `INSERT photos ... ON CONFLICT (project_id, image_id) DO NOTHING RETURNING`; for returned photos: `INSERT items`, `detections`, `class_opinions`, `photo_capture`, `analysis_ledger`, vector rows, `item_state` (fold with no claims), `box_state`, `events`; `COMMIT` | new rows only; a second inserter of the same content waits on the unique index entry, then gets no row back and treats the photo as a duplicate | sees nothing until commit; a photo is stored whole or not at all |
| **Human label, one item** | `INSERT label_operations` (kind `label`); `SELECT items ... FOR UPDATE`; `INSERT claims (class_set)`; read inputs; fold; `UPDATE item_state`; `INSERT events`; `COMMIT` | op row (new), one `items` row | a machine writer on the same item waits on the `items` lock, then folds with the claim on top: the human class stays |
| **Human label, bulk** | one op row (`writing`), then chunks of 1,000 items, each its own transaction: lock items in key order, `INSERT claims` (one multi-row statement), fold each, `UPDATE item_state` (one `UPDATE ... FROM (VALUES ...)`); op `written` at the end, `failed` with the count if a chunk fails | per chunk | a bulk of another person over overlapping items interleaves chunk by chunk; the newer claim (higher `claim_id`) wins per item, deterministically |
| **Undo** (one claim or one op) | `INSERT claims (withdraw)`; then the affected items (by `ix_claims_op` or the one subject) in chunks: lock, fold, update | op row not locked (it is only read); items per chunk | a concurrent undo of the same claim or operation hits `uq_claims_withdraws` / `uq_claims_withdraws_op` (409 `already_undone`) |
| **Box edit** | `SELECT items ... FOR UPDATE`; compare `item_state.region_revision` with `expected_region_revision` (409 `region_revision_conflict` on mismatch, as today); `UPDATE items SET box_seq` for new boxes; `INSERT claims` (box kinds); fold; `UPDATE item_state`, upsert `box_state`; `COMMIT`; after commit, queue the box vector refresh (a result write) | one `items` row | the region worker's write on the same item waits, then folds under the claims |
| **Region-stage result** | lock the batch's items in key order; replace `region_boxes` and `box_texts` of the analyser kind (matching old ids by IoU, new ids from `box_seq`); upsert `item_region_results`, `class_opinions` (combined call); fold; update state rows; `COMMIT` | items of the batch | never conflicts with a person: it writes only results (R5); the fold keeps claims on top |
| **Cluster activation** | `UPDATE cluster_activations SET serving_partition_id = ..., revision = revision + 1 WHERE ... AND revision = :expected` (02d 2.2, OCC); if gate A keeps placement in `item_state`: a job updates placement in chunks under the items lock order; else nothing more | the activation row | readers switch at commit when placement is read from memberships; with the column, they see the new partition chunk by chunk (a read joins the activation's pointer to be exact) |
| **Class merge** | transaction 1: `SELECT ... FROM class_registries FOR UPDATE`; `UPDATE project_classes SET deprecated, merged_into` (02a section 3); `COMMIT` (reads now resolve `merged_into`, so lists are right immediately); then a job: chunks of items with `class_id = :source`, locked in key order, `UPDATE item_state SET class_id = :target` | registry row, then items per chunk | a fold racing the job resolves the claim's class through `merged_into` and writes the target id too; nothing is appended to claims (plan 15 section 5) |
| **Holdout freeze** | op row; `holdout_freezes` row (`uq_holdout_current`); chunks: lock items, `INSERT claims (holdout_frozen, with machine_class)`, fold, update | as bulk label | the snapshot file of today (`holdout.py:290`) stays the audit copy |

Deadlock note: no path locks `items` and then a `photos` row, and no path locks `item_state` rows
without first holding the matching `items` lock, so the order above is total.

## 8. What gate A and gate B can change

| Element | Gate | What may change | Easy to swap because |
|---|---|---|---|
| `item_state` cluster columns, `ix_item_state_cluster`; `box_state` cluster columns | A (plan 15 section 13) | become a side table keyed `(partition_id, subject_key)` swapped at activation (that is `cluster_memberships` itself) | only the cluster read family and the fold's layer 5 read them |
| `item_state.taken_utc`, `taken_local`, `ix_item_state_taken`; `blur_lap_ratio`, `detector_*` copies | A | dropped, replaced by joins, if the joined form meets the page bars | copies written by the fold only |
| `ix_photo_capture_geo`, `ix_text_lines_trgm`, `ix_box_state_text_trgm` | A | another index form, or the search copy | indexes only |
| summary tables for counts (image counts, class counts, cluster cards) | A | created by a later revision if dashboard shapes miss the bars | none exist now |
| `project_indexes`; a change table `item_changes (change_id bigint identity, project_id, item_key, at)` feeding a search copy | A | `project_indexes` deleted in "one store"; the change table created only in "search copy stays" | not created now |
| the approximate index on each vector table; `embedding_spaces.vector_index` | B (plan 15 section 7) | HNSW on `halfvec`, HNSW on bit codes with rescore, or an index outside PostgreSQL fed from the vector tables | the vector tables are the record either way; the index is built from `vector_index` |
| the exact-search threshold | B | a measured row count | a setting, not schema |

## 9. Migration

Current head: `0009_ingest_retry_at` (`src/db/migrations/versions/0009_ingest_retry_at.py:23`).
Revision ids are fixed here and passed with `--rev-id`; one revision-writing lane per step (02d
section 0).

| Revision | Lane (plan 15 section 10) | Creates or changes |
|---|---|---|
| `0010_storage_record` | step 1 | `CREATE EXTENSION IF NOT EXISTS vector` and `pg_trgm` (as `op_owner`; the image must ship pgvector 0.8.7); every table of section 2 with indexes, policies, triggers (`claims_append_only`, `analysers_immutable`); the `label_operations` changes; `dataset_imports.dataset_name`; the `embedding_spaces`, `cluster_memberships`, `near_dup_*`, `holdout_near_dup_moves`, `dataset_import_images` changes; the functions `op_vector_space_create`, `op_vector_space_drop` (question 2) and one vector table per existing space row; grants to `op_app` |
| `0011_ingest_cutover` | step 2 | drops `image_contents` |
| `0012_claims_cutover` | step 3 | drops `label_operation_members`, `near_dup_decisions` |
| `0013_results_cutover` | step 4 | drops `search_projection_tasks` |
| `0014_reads_cutover` | step 5, after gate A | drops `project_indexes` (one store) or creates `item_changes` (search copy) |

Creation order inside `0010`: `analysers`; `photos`, `photo_provenance`; `items`; `entities`
(claims reference it); `claims`; the result tables; `item_state`, `box_state`; the rest of 2.5;
`vector_gaps`; then the vector tables.

No data migration: the 0.7 data is wiped and re-ingested (owner, 2026-11-14); `downgrade()` of each
revision drops what it created (02d section 0). A test `tests/db/test_storage_schema_matches_plan.py`
introspects the migrated database and asserts every table, unique constraint, partial index,
trigger and policy named here, as 02a's schema test does; `src/db/table_use.py` gets a row per new
table (writer, reader, wave), with the phase-5 tables marked "schema-test assertion only" until
phase 5 (02d section 0.1 rule).

## 10. Questions for the coordinator

1. **The core API's data needs a project.** Evidence: the five global indexes have no project
   (`src/clients/opensearch/names.py:18`); plan 15 section 9 says their data "becomes project
   tables"; 02a 1.4 allows NULL-project rows only in a few low-volume tables. Options: (a) a project
   created at bootstrap that the core routes bind (`core_tx` then means "the core project"); (b)
   nullable `project_id` in photos, items, faces, text lines and vectors. Recommendation: (a), with a
   reserved slug added to the `projects` CHECK list; (b) puts a nullable column in every key at 10 M
   rows and keeps a third scope alive.
2. **Creating a table and an index per vector space needs DDL at run time; `op_app` has none.**
   Evidence: plan 15 R6 and section 4.3 ("one table per vector space, created when the space is
   added"); 02a 1.5 (`op_app`: "no DDL"); a plain `CREATE INDEX` may run in a transaction block but
   `CREATE INDEX CONCURRENTLY` may not, so it cannot run inside a function
   ([CREATE INDEX](https://www.postgresql.org/docs/18/sql-createindex.html)).
   Options: (a) `SECURITY DEFINER` functions owned by `op_owner` that build names only from the
   registry row (`SET search_path` fixed, as [CREATE FUNCTION](https://www.postgresql.org/docs/18/sql-createfunction.html)
   "Writing SECURITY DEFINER Functions Safely" says) for create and drop, and a non-concurrent index
   build inside them, which blocks inserts into that one space's table while it builds (minutes to
   hours at 10 M; the space is `building`, so only its own reindex job writes it); (b) the
   reindex job opens one `op_owner` connection for `CREATE INDEX CONCURRENTLY` only; (c) one shared
   table per dimension with a `space_id` column and partial indexes, which still needs DDL for each
   index and makes a drop a 10 M-row delete. Recommendation: (a) for create and drop, and build the
   index in (a) too while the space is `building`; revisit only if gate B needs online rebuilds of a
   `ready` space.
3. **The photo-library plan's tables are mutable rows; plan 15 makes gallery decisions claims.**
   Evidence: [photo_library 00](../photo_library/00_photo_library_plan.md) section 6.1
   (`photo_annotations` with `revision`, `library_operation_members.before jsonb`, `persons.name`);
   plan 15 sections 4.2 and 4.5. This file designs the claim form (section 2.5) and replaces those
   tables. Recommendation: confirm, and have the photo-library plans updated to name these tables
   before phase 5 starts.
4. **"Forget this person" must delete data, R3 says claims are never deleted.** Evidence: photo
   library P2a-2 exit (4) "leaves zero face docs, prototypes, constraints and projections for that
   person" (00 section 2); plan 15 R3. Options: (a) one guarded delete path, like
   `op_project_delete_allows`, that deletes the person's claims and face rows and records only a
   count; (b) keep the claims with the name blanked (an update, also against R3). Recommendation:
   (a); the trigger `claims_append_only` allows it under a transaction-local setting checked against
   an `entity_state.state = 'forgetting'` row, with a test that no other module sets it.
5. **One machine-class order is undefined today; the fold must fix it.** Evidence: section 4.2:
   propagation and an unvalidated import label each overwrite the other, whichever runs last
   (`assist_loop/propagate.py:87`, `dataset_import/item_labels.py:38-40`). Options: import suggestion
   over propagated, or the reverse. Recommendation: import suggestion over propagated (rank 3 over
   4), the same "the item's own answer outranks a carried-over one" rule propagation already applies
   to a dissenting VLM answer.
6. **Plan 15 section 4.4 says a rebuild "empties the table and folds again"; the item identity
   cannot be emptied.** Evidence: vectors, boxes and opinions reference the item key (R6, R8); an
   emptied identity table would orphan 10 M vectors. This file splits identity (`items`, kept and
   garbage-collected by `attach`) from state (`item_state`, emptied and refolded), which keeps the
   plan's sentence true for the state table. Recommendation: confirm the split.
