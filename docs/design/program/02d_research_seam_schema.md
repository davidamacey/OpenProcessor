# 02d: the research-seam tables (part of the PostgreSQL schema)

| Field | Value |
|---|---|
| Status | plan, not started; written 2026-11-04 by the research integration pass on `research/master` |
| Part of | [02a_postgres_schema.md](02a_postgres_schema.md) (its section 1 conventions apply to every table here: `uuidv7()` keys, `timestamptz`, closed sets as `text` + `CHECK` with one Python source, `jsonb` only for pydantic-validated bodies, row-level security on every `project_id` table, `*_by` actor columns) |
| Sources | the section 5 seams of the seven research plans: [GPU pipeline](../research_gpu_pipeline_plan.md), [clustering](../research_clustering_plan.md), [embeddings and search](../research_embeddings_search_plan.md), [near-duplicates](../research_near_duplicates_plan.md), [spatial and attributes](../research_spatial_attributes_plan.md), [registration-plate text](../research_plate_text_plan.md), [auto-label assistant](../research_auto_label_assist_plan.md). Each table names its plan; the plan holds the reasoning, this file holds the contract |
| Decided | approved by the owner 2026-11-04: every seam goes in now (RD-1 and review Q-A: the review's deferrals are rejected), the plate tables get their own revision in W5e (RD-4), cluster memberships live in PostgreSQL (RD-23), GPS coordinates are stored behind a per-project policy (RD-51 reversed). The review's correctness findings RR-2, RR-3, RR-17, RR-18 are applied below ([00e_research_review.md](00e_research_review.md)) |
| Cross-topic rules | [00b_traceability.md](00b_traceability.md) section 5.1 (X-1..X-20): why these are separate tables with shared patterns |

Why a separate file: 02a describes the control plane that B2 moves; this file describes the tables
the research topics need so that each later winner lands as data or a new method value, not as a
schema redesign. One concern per file (the ~700-line guide); the tests treat both files as one
contract (`tests/db/test_schema_matches_plan.py` reads 02a, `tests/db/test_research_schema_matches_plan.py`
reads this file).

---

## 0. Revision chain and who writes it

One Alembic head at all times (`tests/db/test_migrations.py`). Revision ids are fixed here and passed
with `--rev-id`, so two lanes never guess the chain:

| Revision | Lane, wave | Creates or changes | Rule |
|---|---|---|---|
| `0001_control_plane` | 1A `b2/db-core`, W2.1 | every table of 02a, **including the in-place research changes listed in section 1** (columns and `CHECK` values on 02a tables, the `gpu_devices` table) | a change to a table 0001 creates is made in 0001 |
| `0002_research_seams` | 1F `b2/db-research-seams`, W2.1 (cut after 1A merged; [03a](03a_b2_w1_queue_infra.md) lane 1F) | section 2: serving reports, clustering S-core, the embedding registry, the auto-label loop record | only new tables; `down_revision = '0001_control_plane'` |
| `0003_near_duplicates` | ND-A `b2/near-duplicates`, W2.3b ([03b](03b_b2_w2_job_families.md) lane ND-A) | section 3: the near-duplicate tables, three `holdout_freezes` columns, the `near_dup_profile` kind and `near_dup` axis, the `near_dup_run` projection kind | the only revision-writing lane of W2.3b (W2.3a writes none) |
| `0004_region_text` | `w5e/region-text-store`, W5e ([03g](03g_w5e_plate_text.md)) | section 4: the registration-plate text tables and the `region_text_consensus_marker` state key | approved (RD-4) |

A wave that needs a schema change has exactly one revision-writing lane. Downgrade is tested to base
and back (1A-3), so every revision has a real `downgrade()`.

## 1. In-place changes to 02a tables (made in 0001 by lane 1A; the DDL is edited in 02a itself)

| 02a table | Change | Source seam | 0.7.0 writer and reader |
|---|---|---|---|
| `deployment_settings` | key `active_serving_profile` (`{config_id, revision}`, OCC on `revision`) | GPU S-5, resolution X-2 | written by 2C's activate route for the deployment document; read by the Triton router (4B) and `openprocessor gpu plan` |
| `config_documents` | kinds `serving_profile` (deployment), `cluster_profile` (project), `assist_loop_preset` (project); the scope check becomes `CHECK ((kind IN ('vlm_endpoint','serving_profile')) = (project_id IS NULL))` | GPU S-5; clustering 5.4; auto-label 5.7 | 2C (body models), 4F, 4D |
| `config_activations` | axes `clustering_item`, `clustering_region_box`, `assist_loop_preset`, with the pairs `('clustering_item','cluster_profile')`, `('clustering_region_box','cluster_profile')`, `('assist_loop_preset','assist_loop_preset')` | clustering 5.4; auto-label 5.7 | 2C, read by 4F and 4D |
| `project_settings` | `search jsonb NULL` (`SearchSettings`: default `group_by`, RRF `rank_constant`, collapse over-fetch, default fused spaces) | embeddings 5.6 | 2C, read by ES-C's search builder |
| `project_state` | the key list becomes `('cluster_refresh','region_drain','region_dependency_health')`: `region_cluster_refine_marker` and `region_cluster_partition_marker` are never written (lane 4F replaces region clustering with partitions in the same release), `last_clustering` stays a file until 4F deletes `last_run.py` | clustering 5.6, resolution X-11 | 2C (`cluster_refresh`, interim body), 2D (`region_drain`, `region_dependency_health`); 4F re-defines the `cluster_refresh` body as the beat's scheduling clock (deferral start, last attempt, backoff) |
| `search_projection_tasks` | kind `cluster_partition` (payload `{target, level, embedding_field, partition_id}`) | clustering 5.1 (activation) | 4F's activation transaction; consumed by `cluster_projection_sync` |
| `jobs` | `retry_count integer NOT NULL DEFAULT 0`; `pause_reason text NULL CHECK (pause_reason IN ('user','audit','budget','project_paused','vlm_unavailable'))`; `CHECK (status <> 'paused' OR pause_reason IS NOT NULL)` | auto-label 5.2 (G-1) | 3A (claim, pause, resume, reaper) |
| `job_attempts` | outcome gains `yielded` | auto-label 5.2 (G-2) | 3A wrapper |
| `gpu_devices` (new, in 0001 because `gpu_reservations` references it) | see 02a section 5 | GPU S-6 | GPU workers (start and heartbeat), `op.periodic.serving_inventory` and the reservation insert (review RR-9); read by `openprocessor gpu plan`, reservations and the serving planner |
| `gpu_reservations` | primary key `(gpu_uuid)` referencing `gpu_devices` | GPU S-7, resolution X-6 | 6A (arbiter), 5A (training) |
| `training_runs` | `gpu_ids integer[]` becomes `gpu_uuids text[]` | resolution X-6 | 5A |
| `model_engines` | input and output contracts, build profile, settings, stamp, hashes, measured device memory, batch limits, CUDA graphs, precision `CHECK`, parity reference; unique key `(triton_model_name, trt_version, gpu_arch, build_profile)` | GPU S-1..S-4 | 5B and 5D write; read by the CPU-route contract test, `gpu plan`, ES-A's input-spec check |
| `project_settings` | `location_policy text NOT NULL DEFAULT 'store' CHECK (location_policy IN ('drop','store_coarse','store'))` (owner 2026-11-04, RD-51 reversed: store GPS coordinates now, default keep); the 0.6.0 upgrade writes `store` for every project | spatial 5.1 | 2C (settings model and route, with `expected_revision`); read by the W4 capture lane's writer and by the one visibility helper |

The coordinates themselves are image search data, so they live where every image field lives, in
the OpenSearch images index (`capture.location` `geo_point`, `capture.altitude_m`,
`capture.location_accuracy_m`, `capture.heading_deg`, `capture.location_precision`), written by the
W4 capture lane under this policy (`store_coarse` rounds to about 1 km); PostgreSQL has no per-image
table to hold them (D-40). They are never copied to items, labels, manifests, training files or a VLM
call; switching a project to `drop` purges them (a step of `source_backfill`, counts-only audit).

## 2. Revision `0002_research_seams` (lane 1F, W2.1)

Creation order: `serving_reports`; `cluster_calibrations`, `cluster_partitions`, `cluster_identities`,
`partition_clusters`, `cluster_memberships`, `cluster_activations`, `cluster_activation_events`,
`cluster_backend_validations`; `embedding_model_versions`, `embedding_spaces`, `embedding_reindex_runs`,
`embedding_evaluation_runs`; the `assist_loop_*` tables, then `vlm_usage_hourly`.

### 2.1 Serving reports (GPU S-14)

```sql
CREATE TABLE serving_reports (               -- deployment data, no RLS; written by op.periodic.serving_inventory
  server      text NOT NULL,                 -- a server name of the active serving profile
  model       text NOT NULL,
  version     integer NOT NULL,
  state       text NOT NULL CHECK (state IN ('ready','unavailable','loading','unloading')),
  instances   integer NULL,
  device_mib  integer NULL,
  seen_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (server, model, version)
);
```

Reader in 0.7.0: the Triton router of B4 4B (a server whose report lacks the model gets no call).

### 2.2 Clustering S-core (clustering plan 5.1, with the consistency fixes of X-1 and X-5)

```sql
CREATE TABLE cluster_calibrations (           -- immutable (trigger cluster_calibrations_immutable;
                                              -- DELETE only by cascade or op_project_delete_allows, RN-1)
  id               uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id       uuid NULL REFERENCES projects(id) ON DELETE CASCADE,   -- NULL: deployment default
  target           text NOT NULL CHECK (target IN ('item','region_box','face','image')),
  level            text NOT NULL CHECK (level IN ('class','identity','scene')),
  embedding_field  text NOT NULL,
  embedding_model  text NOT NULL,             -- X-5: the space_key for project spaces, the catalog key for core
  preprocess_id    text NOT NULL,             -- X-5: embedding_model_versions.input_spec_sha256
  kind             text NOT NULL CHECK (kind IN ('cut','cpm_gamma','assign_bands','core_point')),
  fitted_on        text NOT NULL,             -- 'pin:<manifest_sha256>' or 'project:<id>@holdout:<freeze id>'
  body             jsonb NOT NULL,            -- ClusterCalibrationBody (thresholds, isotonic knots, bands)
  dev_metrics      jsonb NOT NULL,
  created_at       timestamptz NOT NULL DEFAULT now(),
  created_by       text NOT NULL
);

CREATE TABLE cluster_partitions (             -- one clustering run's output; immutable after 'ready' except status
  id                  uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id          uuid NULL REFERENCES projects(id) ON DELETE CASCADE,  -- NULL: core faces (persons_cluster)
  target              text NOT NULL CHECK (target IN ('item','region_box','face','image')),
  level               text NOT NULL CHECK (level IN ('class','identity','scene')),
  embedding_field     text NOT NULL,
  embedding_model     text NOT NULL,          -- X-5
  preprocess_id       text NOT NULL,          -- X-5
  method              text NOT NULL CHECK (method IN ('ivf','minibatch_kmeans','union_find','detector_class','edit')),
                                              -- research values (graph_hac, leiden_cpm, core_point, hdbscan) are added
                                              -- by the landing lane of the method that passes (no unused value now)
  method_version      text NOT NULL,
  profile_config_id   uuid NULL,
  profile_revision    integer NULL,
  params              jsonb NOT NULL,         -- the method's pydantic params model
  calibration_id      uuid NULL REFERENCES cluster_calibrations(id),   -- NO ACTION (02a 1.2)
  pool_filter         jsonb NOT NULL,         -- the residual gate as data
  input_digest        text NOT NULL,          -- sha256 of the sorted (item_key, vector version) list
  seed                bigint NOT NULL,
  backend             text NOT NULL CHECK (backend IN ('cpu','gpu')),
  backend_detail      jsonb NOT NULL,
  parent_partition_id uuid NULL REFERENCES cluster_partitions(id) ON DELETE SET NULL,
  job_id              uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  status              text NOT NULL CHECK (status IN ('building','ready','failed','discarded')),
  n_items integer NULL, n_clusters integer NULL, n_review integer NULL, n_unassigned integer NULL,
  metrics             jsonb NOT NULL DEFAULT '{}',
  diff                jsonb NULL,             -- MONIC transitions against the parent, anchored moves
  stage_durations     jsonb NOT NULL DEFAULT '{}',
  peak_rss_mb integer NULL, peak_vram_mb integer NULL,
  created_at          timestamptz NOT NULL DEFAULT now(),
  finished_at         timestamptz NULL,
  created_by          text NOT NULL,
  UNIQUE (id, project_id),                    -- target of the child composite FKs
  FOREIGN KEY (profile_config_id, profile_revision) REFERENCES config_revisions(config_id, revision),  -- X-1: NO ACTION
  CHECK ((profile_config_id IS NULL) = (profile_revision IS NULL))
);
CREATE INDEX ix_cluster_partitions_recent ON cluster_partitions (project_id, target, level, created_at DESC);
-- trigger cluster_partitions_ready_immutable: once status = 'ready', only status may change, and
-- job_id and parent_partition_id may change to NULL (review RR-2: the ON DELETE SET NULL writes of
-- job pruning and of partition retention are UPDATEs that the trigger must let through); a DELETE of a
-- 'ready' partition passes only by cascade or under op_project_delete_allows (review RN-1)

CREATE TABLE cluster_identities (             -- stable identity of a cluster across partitions
  cluster_uid        uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id         uuid NULL REFERENCES projects(id) ON DELETE CASCADE,
  target text NOT NULL CHECK (target IN ('item','region_box','face','image')),   -- review RR-17
  level  text NOT NULL CHECK (level IN ('class','identity','scene')),
  name               text NULL,
  name_source        text NULL CHECK (name_source IN ('human','import','vlm')),
  anchored_at        timestamptz NULL,
  anchored_by        text NULL,
  promoted_class_id  integer NULL,
  first_partition_id uuid NULL REFERENCES cluster_partitions(id) ON DELETE SET NULL,
  last_partition_id  uuid NULL REFERENCES cluster_partitions(id) ON DELETE SET NULL,
  revision           bigint NOT NULL DEFAULT 1,
  FOREIGN KEY (project_id, promoted_class_id) REFERENCES project_classes(project_id, class_id),  -- added here (X-5)
  CHECK ((anchored_at IS NULL) = (anchored_by IS NULL))
);

CREATE TABLE partition_clusters (
  partition_id     uuid NOT NULL REFERENCES cluster_partitions(id) ON DELETE CASCADE,
  cluster_uid      uuid NOT NULL REFERENCES cluster_identities(cluster_uid) ON DELETE CASCADE,
  project_id       uuid NULL,
  display_no       integer NOT NULL,
  parent_uid       uuid NULL,
  transition       text NOT NULL CHECK (transition IN ('new','survived','split','merged','absorbed')),
  predecessor_uids uuid[] NOT NULL DEFAULT '{}',
  size             integer NOT NULL CHECK (size > 0),
  dispersion       real NULL,
  prototypes       jsonb NOT NULL DEFAULT '[]',
  PRIMARY KEY (partition_id, cluster_uid),
  UNIQUE (partition_id, display_no),
  FOREIGN KEY (partition_id, project_id) REFERENCES cluster_partitions(id, project_id) ON DELETE CASCADE
);

CREATE TABLE cluster_memberships (            -- bulk-written with COPY by the run
  partition_id  uuid NOT NULL REFERENCES cluster_partitions(id) ON DELETE CASCADE,  -- added (X-1: the composite FK
                                                                                    -- is not checked when project_id is NULL)
  project_id    uuid NULL,
  item_key      text NOT NULL,                -- crop_id; crop_id || ':' || box_id for boxes; face id
  cluster_uid   uuid NULL,
  state         text NOT NULL CHECK (state IN ('assigned','review','unassigned')),
  score         real NULL,
  margin        real NULL,
  PRIMARY KEY (partition_id, item_key),
  FOREIGN KEY (partition_id, project_id) REFERENCES cluster_partitions(id, project_id) ON DELETE CASCADE,
  CHECK ((state = 'unassigned') = (cluster_uid IS NULL))
);
CREATE INDEX ix_cluster_memberships_cluster ON cluster_memberships (partition_id, cluster_uid);
-- trigger cluster_children_ready_immutable on partition_clusters and cluster_memberships:
-- UPDATE or DELETE raises while the parent partition is 'ready'; a DELETE passes when the parent is
-- gone (cascade), no longer 'ready' (retention sets 'discarded' first, then deletes in batches), or
-- op_project_delete_allows(project_id) is true (the project_delete job; 02a section 1.2, review RN-1)
-- trigger cluster_children_project_from_parent (review RR-18): BEFORE INSERT copies project_id from
-- the partition, so a project partition can never get a NULL-project membership (the composite key
-- is MATCH SIMPLE and skips NULLs); one lookup per COPY batch

CREATE TABLE cluster_activations (            -- the served partition per (project, target, level, field); OCC on revision
  id                   uuid PRIMARY KEY DEFAULT uuidv7(),   -- review RR-17: the ORM needs a key; project_id is nullable
  project_id           uuid NULL REFERENCES projects(id) ON DELETE CASCADE,
  target               text NOT NULL CHECK (target IN ('item','region_box','face','image')),
  level                text NOT NULL CHECK (level IN ('class','identity','scene')),
  embedding_field      text NOT NULL,
  serving_partition_id uuid NOT NULL REFERENCES cluster_partitions(id),   -- NO ACTION (02a 1.2)
  pending_partition_id uuid NULL REFERENCES cluster_partitions(id),
  revision             bigint NOT NULL DEFAULT 1,
  updated_at           timestamptz NOT NULL DEFAULT now(),
  updated_by           text NOT NULL,
  UNIQUE NULLS NOT DISTINCT (project_id, target, level, embedding_field)
);

CREATE TABLE cluster_activation_events (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  project_id uuid NULL REFERENCES projects(id) ON DELETE CASCADE,
  target text NOT NULL, level text NOT NULL, embedding_field text NOT NULL,
  from_partition_id uuid NULL, to_partition_id uuid NOT NULL,
  reason text NOT NULL CHECK (reason IN ('auto','human','rollback','edit','upgrade')),
  at timestamptz NOT NULL DEFAULT now(), by text NOT NULL
);

CREATE TABLE cluster_backend_validations (    -- deployment data, no RLS, written under deployment_tx
  id               uuid PRIMARY KEY DEFAULT uuidv7(),
  method           text NOT NULL,
  backend_versions jsonb NOT NULL,
  embedding_field  text NOT NULL,
  dataset_pin      text NOT NULL,
  ari_vs_cpu       real NOT NULL, purity_delta real NOT NULL, graph_recall real NULL,
  passed           boolean NOT NULL,
  validated_at     timestamptz NOT NULL DEFAULT now(),
  validated_by     text NOT NULL
);
```

Reader of `cluster_backend_validations` in 0.7.0: the `cluster_run` start refuses `backend = 'gpu'`
without a passing row (the agreement gate as a constraint). Rules (activation transaction, id
carry-over, auto-activation, retention): clustering plan section 5.1, implemented by lane 4F. The
closed lists (`target`, `level`, `method`, `state`, `transition`) live in `src/domain/clustering_vocab.py`
(lane 1F) and a test asserts each `CHECK` equals it.

### 2.3 Embedding registry (embeddings plan 5.2)

```sql
CREATE TABLE embedding_model_versions (       -- deployment catalog; no project_id, no RLS
  id                   uuid PRIMARY KEY DEFAULT uuidv7(),
  model_key            text NOT NULL CHECK (model_key ~ '^[a-z][a-z0-9_]{1,39}$'),
  version              integer NOT NULL CHECK (version >= 1),
  modalities           text[] NOT NULL CHECK (modalities <@ ARRAY['image','text'] AND cardinality(modalities) >= 1),
  native_dim           integer NOT NULL CHECK (native_dim BETWEEN 8 AND 4096),
  matryoshka_dims      integer[] NOT NULL DEFAULT '{}',
  distance             text NOT NULL CHECK (distance IN ('cosine')),
  input_spec           jsonb NOT NULL,        -- EncoderInputSpec: {image: EngineInputContract, text: {context_length, prefixes}} (X-5)
  input_spec_sha256    text NOT NULL,         -- also the preprocess_id that clustering and near-duplicate runs record
  weights_ref          text NOT NULL,
  weights_sha256       text NOT NULL,
  precision            text NOT NULL CHECK (precision IN ('fp32','bf16','fp16','int8')),
  triton_image_model   text NULL,
  triton_text_model    text NULL,
  engine_parity_min_cos real NULL CHECK (engine_parity_min_cos BETWEEN 0 AND 1),   -- written by the W3 gate G7
  licence              text NOT NULL,
  status               text NOT NULL CHECK (status IN ('available','deprecated','retired')),
  created_at           timestamptz NOT NULL DEFAULT now(),
  created_by           text NOT NULL,
  UNIQUE (model_key, version),
  CHECK ((triton_image_model IS NOT NULL) = ('image' = ANY(modalities))),
  CHECK ((triton_text_model  IS NOT NULL) = ('text'  = ANY(modalities)))
);
-- trigger embedding_model_versions_immutable: BEFORE UPDATE raises unless only status or
-- engine_parity_min_cos changes (the W3 gate records parity once; a lower parity is a new version)

CREATE TABLE embedding_spaces (
  id                uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id        uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  space_key         text NOT NULL CHECK (space_key ~ '^[a-z][a-z0-9_]{2,47}$'),
  model_version_id  uuid NOT NULL REFERENCES embedding_model_versions(id) ON DELETE RESTRICT,
  target            text NOT NULL CHECK (target IN ('item','box','frame','item_text')),
  dim               integer NOT NULL CHECK (dim BETWEEN 8 AND 4096),
  role              text NOT NULL CHECK (role IN ('primary','optional','candidate')),
  status            text NOT NULL CHECK (status IN ('pending','building','ready','retiring','retired','failed')),
  index_role        text NOT NULL CHECK (index_role IN ('images','items')),
  vector_field      text NOT NULL,
  state_field       text NULL,
  nested_path       text NULL,
  knn_method        jsonb NOT NULL,           -- KnnMethodSpec, read from the live mapping on upgrade
  revision          bigint NOT NULL DEFAULT 1,
  built_by_job_id   uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  ready_at          timestamptz NULL,
  retired_at        timestamptz NULL,
  created_at        timestamptz NOT NULL DEFAULT now(),
  created_by        text NOT NULL,
  UNIQUE (project_id, target, space_key),
  UNIQUE (project_id, index_role, vector_field),
  CHECK ((target = 'box') = (nested_path IS NOT NULL)),
  CHECK ((target = 'frame') = (index_role = 'images')),
  CHECK (role <> 'primary' OR status = 'ready')
);
CREATE UNIQUE INDEX uq_embedding_spaces_one_primary ON embedding_spaces (project_id, target) WHERE role = 'primary';

CREATE TABLE embedding_reindex_runs (
  id             uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id     uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  space_id       uuid NOT NULL REFERENCES embedding_spaces(id) ON DELETE CASCADE,
  job_id         uuid NOT NULL UNIQUE REFERENCES jobs(id),   -- NO ACTION; jobs keep-list (02a section 9)
  mode           text NOT NULL CHECK (mode IN ('build','backfill','clear')),
  scope          jsonb NOT NULL DEFAULT '{}',
  docs_total     bigint NULL,
  docs_done      bigint NOT NULL DEFAULT 0,
  docs_failed    bigint NOT NULL DEFAULT 0,
  docs_skipped   bigint NOT NULL DEFAULT 0,
  bytes_estimate bigint NULL,
  started_at     timestamptz NULL,
  finished_at    timestamptz NULL
);

CREATE TABLE embedding_evaluation_runs (
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id         uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id             uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  baseline_space_id  uuid NOT NULL REFERENCES embedding_spaces(id) ON DELETE CASCADE,
  candidate_space_id uuid NOT NULL REFERENCES embedding_spaces(id) ON DELETE CASCADE,
  protocol           text NOT NULL CHECK (protocol IN ('label_consistency_v1','query_set_v1')),
  sample_spec        jsonb NOT NULL,
  metrics            jsonb NOT NULL,
  verdict            text NOT NULL CHECK (verdict IN ('better','tie','worse','inconclusive')),
  created_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,
  CHECK (baseline_space_id <> candidate_space_id)
);
```

Rows at creation and upgrade (catalog sync in `db-migrate`, three default spaces in the project
create transaction, spaces per upgraded project with `knn_method` read from the live mapping):
embeddings plan 5.2, lanes ES-A, 2A and 2E ([03d](03d_b2_s2_control_data.md)).

### 2.4 Auto-label loop record (auto-label plan 5.1; fixes gap G-3)

The DDL is the auto-label plan's section 5.1 **as written**, with one change: `assist_loop_runs`
takes the composite profile reference of X-1 instead of a `config_documents` reference with
`SET NULL`, and its job reference is `NO ACTION` instead of `RESTRICT` (02a section 1.2):

```sql
CREATE TABLE assist_loop_runs (
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),     -- the loop_id on the wire
  project_id         uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id             uuid NOT NULL UNIQUE REFERENCES jobs(id),   -- NO ACTION; jobs keep-list
  preset_config_id   uuid NULL,
  preset_revision    integer NULL,
  request            jsonb NOT NULL,
  strategy           text NOT NULL CHECK (strategy IN ('cluster_sampled')),
  seed               bigint NOT NULL,
  pinned             jsonb NOT NULL,        -- vlm, prompt pack, audit endpoint, partition {partition_id, digest, snapshot_at}
  stop_reason        text NULL CHECK (stop_reason IN ('done','budget_crops','budget_requests','budget_tokens','budget_time','vlm_failed')),
  created_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,
  created_by_user_id uuid NULL,
  FOREIGN KEY (preset_config_id, preset_revision) REFERENCES config_revisions(config_id, revision),
  CHECK ((preset_config_id IS NULL) = (preset_revision IS NULL))
);
```

and then, unchanged from the auto-label plan section 5.1: `assist_loop_iterations`,
`assist_loop_nodes` (members snapshot as a storage blob per node), `assist_loop_selections`,
`assist_loop_proposals`, `assist_loop_reviews` (created now so the review data model is fixed before
its v0.8.0 routes; the 0.7.0 reader is the blind-audit `CHECK` test and the metrics function of
lane 4D), `assist_loop_budgets` (with the over-reserve `CHECK`), `assist_loop_budget_ledger` (fenced
by `fencing_token`), `vlm_usage_hourly` (every VLM caller). RLS: `assist_loop_runs` and
`vlm_usage_hourly` carry `project_id` (direct policy); the other loop tables reach the project
through `run_id` with the `EXISTS` policy of 02a section 5 (they are bounded by the run's budget, so
the per-row `EXISTS` is acceptable; contrast X-1's rule for unbounded membership tables). The
`pinned.partition` value names a `cluster_partitions` row (X-1: the loop reads
`(partition_id, cluster_uid)`, never an integer `cluster_id`).

## 3. Revision `0003_near_duplicates` (lane ND-A, W2.3b)

The DDL is the near-duplicate plan's section 5.1, with three changes (X-1, X-5):

1. (and review RR-18: the same `BEFORE INSERT` trigger copies `project_id` from the run into groups and
   members) `near_dup_runs` gains `UNIQUE (id, project_id)`; `near_dup_groups` and `near_dup_members` gain a
   denormalized `project_id uuid NULL` with `FOREIGN KEY (run_id, project_id) REFERENCES
   near_dup_runs(id, project_id) ON DELETE CASCADE` (the single-column `run_id` foreign key stays,
   because a composite key is not checked for the NULL-project `core` level). Their RLS policy is
   then the direct constant-InitPlan form of 02a section 1.4, not an `EXISTS` per row: a full build
   writes one members row per image and per item, the same growth as `cluster_memberships`.
2. `near_dup_runs.space` holds the registry `space_key` for the `image` and `item` levels (for
   example `pe_core_l14_336_v1`) and `mobileclip2_s2_v1` for `core`; `near_dup_edges.space` the same.
3. The profile reference keeps the plan's composite foreign key, without `ON DELETE RESTRICT`
   (NO ACTION, 02a section 1.2), and so does every reference between two rows that one project
   delete cascades to.

In the same revision:

```sql
ALTER TABLE holdout_freezes
  ADD COLUMN near_dup_image_run_id uuid NULL REFERENCES near_dup_runs(id) ON DELETE SET NULL,
  ADD COLUMN near_dup_item_run_id  uuid NULL REFERENCES near_dup_runs(id) ON DELETE SET NULL,
  ADD COLUMN moved_by_near_dup     integer NOT NULL DEFAULT 0;
-- config_documents.kind CHECK gains 'near_dup_profile'; config_activations.axis gains 'near_dup'
-- with the pair ('near_dup','near_dup_profile'); search_projection_tasks.kind gains 'near_dup_run'
```

Tables: `image_contents` (the exact key: `UNIQUE NULLS NOT DISTINCT (project_id, content_sha256)`;
the insert is the duplicate check, replacing the `imohash` lookup, pending RD-43), `near_dup_runs`
(one `current` and one `building` per project and level by partial unique indexes),
`near_dup_groups`, `near_dup_members`, `near_dup_edges`, `near_dup_decisions`, `near_dup_pending`,
`holdout_near_dup_moves`.

## 4. Revision `0004_region_text` (lane `w5e/region-text-store`, W5e; approved, RD-4)

The DDL is the registration-plate plan's section 5.2, with one change: `region_text_runs` refers to
its profile by `(profile_config_id, profile_revision) REFERENCES config_revisions(config_id,
revision)` (X-1) instead of a `config_documents` reference with `SET NULL`. Tables:
`identity_text_policies` (mode, salt, key version, raw retention; the salt is read only through
`identity_text_keys.project_key()`), `region_text_runs`, `region_text_consensus_groups`
(`basis IN ('near_dup','capture_window','human')`: X-16 says the `near_dup` basis reads the image
level `dedup` groups of the current near-duplicate run, and `capture_window` reads the
`capture_taken_utc` item field of the W4 capture lane), `region_text_consensus_members`. The same
revision adds `region_text_consensus_marker` to the `project_state.key` `CHECK`. **No plate text is
stored in PostgreSQL** (a schema test asserts these tables hold no text-valued column besides the
listed codes).

## 5. Reserved, not created in v0.7.0 (the DDL exists so the later lane does not redesign it)

| Tables or values | Plan and section | Created by | Adjustments to apply when they land |
|---|---|---|---|
| `cluster_constraints`, `cluster_suggestions` | clustering 5.1 (S-later) | research wave RW-3 lane L4, on a CL-5 PASS | `source` values `identity_text` (plate links) and `near_dup` (leak groups as must-link, X-1) |
| `cluster_verification_runs`, `cluster_verdicts` | clustering 5.1 (S-later) | v0.8.0 with the Cropwright review screens (L6) | none |
| research `method` values (`graph_hac`, `leiden_cpm`, `core_point`, `hdbscan`) | clustering 5.1 | the method's landing lane (L3, L5) | one `ALTER ... CHECK` per method |
| `attribute_signals`, `attribute_signal_versions`, `project_attribute_calibrations`; kind `attribute_profile`, axis `attributes` | spatial 5.3 | research wave RW-5 lane L-A1, after a P-2 or P-3 PASS | none |
| `scenes`, `scene_versions`, `scene_members`, `scene_corrections`, `place_pair_evidence`, `reconstruction_runs`, `reconstruction_artifacts`, `camera_poses`; kind `spatial_profile`, axis `spatial`; projection kind `scene_membership`; queue `gpu_spatial` | spatial 5.4 | the spatial design doc an S1 GO triggers (L-SP1) | `scene_versions.manifest_sha256` hashes member image ids plus `content_sha256` (X-4), not `imohash` |
| (no longer reserved) `project_settings.location_policy`, the coordinates and the location purge | spatial 5.1, L-T1 | now in 0001 (section 1) and the W4 capture lane (owner, RD-51 reversed); only the timeline view stays a v0.8.0 lane | see section 1 |

## 6. Tests (lane 1F unless named)

| Test | Asserts |
|---|---|
| `tests/db/test_research_schema_matches_plan.py` | every table, unique constraint, partial index, foreign key, trigger and policy of sections 2-4 present after `upgrade head` (sections 3 and 4 entries are added by ND-A and the W5e store lane) |
| `tests/db/test_cluster_partition_immutable.py` | updating a `ready` partition's memberships or clusters raises; a cascade delete of a discarded partition passes; review RR-2: pruning a `cluster_run` job that a ready partition references passes (the `SET NULL` update is allowed); deleting a ready parent of a ready child passes; retention of a superseded `ready` partition passes by setting it `discarded` first and then batch-deleting its children (review RN-1); deleting a project that holds ready partitions passes, both through the `project_delete` job's batched delete under `op_project_delete_allows` and by a direct `DELETE FROM projects` (cascade); a child DELETE of a `ready` partition without the setting raises |
| `tests/db/test_children_project_from_parent.py` (review RR-18) | a membership or near-duplicate member inserted with `project_id` NULL into a project partition or run gets the parent's project |
| `tests/db/test_cluster_memberships_fk_null_project.py` | a membership of a NULL-project partition that does not exist is refused (the single-column foreign key, X-1) |
| `tests/db/test_embedding_registry_schema.py` | a second primary for one target is a unique violation; the model-version trigger refuses an `input_spec` update and allows `engine_parity_min_cos` once |
| `tests/db/test_assist_loop_schema.py` | the auto-label plan's checks: `CHECK`s, cascades, RLS through `run_id`, the budget `CHECK` refuses an over-reserve, the blind-audit row refuses a shown class |
| `tests/db/test_profile_fk_cascade.py` | review RR-6: the referencing pairs are **enumerated from the catalog** (every foreign key between two tables one project delete reaches, so a new pair cannot be missed); with every pair populated, the `project_delete` job's batched delete (RR-7) and a direct `DELETE FROM projects` both pass; a pair that fails is declared `DEFERRABLE INITIALLY DEFERRED` in its revision; deleting a referenced profile document alone is refused |
| `tests/db/test_row_level_security.py` (1A) | iterates **every** table with a `project_id` column found by introspection, so the tables of this file are covered without a second list |
| `tests/domain/test_clustering_vocab_matches_check.py` | each closed list equals its `CHECK` |
