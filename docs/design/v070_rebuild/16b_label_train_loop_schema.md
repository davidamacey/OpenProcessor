# 16b: the cluster, label and train loop, schema and correctness under concurrency

| Field | Value |
|---|---|
| Status | plan, revision 2 (2026-11-16), companion of [16](16_label_train_loop_plan.md) (the person's loop, what the data says, Part A and Part B) and [16a](16a_label_train_loop_slices.md) (the slices). Nothing here is built |
| What a lane takes from here | the exact DDL of its revision, the lock-order rows, the erasure rows and the concurrency rules its tests prove |
| Binding | [15a](15a_storage_schema.md) (conventions section 1, claims section 3, fold section 4, transactions section 7, erasure section 8), [02d](02d_research_seam_schema.md) section 2.2 (the cluster tables and their triggers), [02a](02a_postgres_schema.md) sections 1.4 (row security) and 3 (class registry). Wipe and re-ingest, no data migration (18 section 0): every statement below runs on empty tables |

Row security: every new table with `project_id` gets exactly the three statements the 0002 revision
uses for its direct-policy tables (`src/db/migrations/schema_0002/security.py`, `_enable` and
`_policy`):

```sql
ALTER TABLE <t> ENABLE ROW LEVEL SECURITY;
ALTER TABLE <t> FORCE ROW LEVEL SECURITY;
CREATE POLICY p_<t>_project ON <t> FOR ALL
  USING ((SELECT op_scope_is_deployment()) OR project_id = (SELECT op_scope_project()))
  WITH CHECK ((SELECT op_scope_is_deployment()) OR project_id = (SELECT op_scope_project()));
```

Written "RLS(<t>)" below; `tests/db/test_row_level_security.py` already iterates every `project_id`
table. Indexes follow 15a section 1's rule (every foreign key whose referenced row can be deleted
gets `ix_<table>_<column>`).

## 1. Part A: the schema the parity slices need

### 1.1 Revision `0016_cluster_cutover` (18b R8, slice R8-1 of 16a)

15a section 2.3 already turns `cluster_memberships.item_key text` into `subject_key bigint` with
primary key `(partition_id, subject_key)` and no foreign key. This revision adds what placement,
erasure and the record's reads need:

```sql
-- placement (16 section 2.2, row "New photos"): a membership row says who wrote it, set at insert, never updated
ALTER TABLE cluster_memberships
  ADD COLUMN assigned_by text NOT NULL
      CONSTRAINT ck_cluster_memberships_assigned_by CHECK (assigned_by IN ('run','incremental'));
ALTER TABLE cluster_memberships SET (autovacuum_vacuum_insert_scale_factor = 0.01);   -- insert-mostly, as claims

-- the centroid as a column (today a jsonb list of 1,024 rounded floats inside prototypes); f32, no dimension:
-- the partition's space fixes it, and placement compares in f32 as the fit did (no halfvec rounding)
ALTER TABLE partition_clusters DROP COLUMN prototypes,
  ADD COLUMN centroid vector NOT NULL;
  -- prototypes held {vector, count, item_key}: count is the size column, the medoid item_key had no reader
  -- (cluster_representatives.py reads the top members by score) and is not kept, so no item key hides in jsonb
```

Not changed, and why:

- `cluster_partitions.method` keeps `ck_cluster_partitions_method` as 02d wrote it, except
  `detector_class`, which no code path writes (survey: `run.py:46` `METHODS = ('ivf',
  'minibatch_kmeans')`) and leaves by 02d's own rule ("no unused value"):

  ```sql
  ALTER TABLE cluster_partitions DROP CONSTRAINT ck_cluster_partitions_method,
    ADD CONSTRAINT ck_cluster_partitions_method
        CHECK (method IN ('ivf','minibatch_kmeans','union_find','edit'));
  ```

  If R8-1's box-cluster bar passes (16 section 2.2, row "Region boxes"), the same statement drops
  `minibatch_kmeans` as well and `src/domain/clustering_vocab.py` follows in the same commit.
- `n_items`, `n_clusters`, `partition_clusters.size`, `dispersion`, `input_digest` describe **the
  fit** and stay immutable after `ready` as 02d says. Live sizes and labelled counts of a serving
  group are read at request time: `cluster_memberships` of the serving partition joined to
  `item_state` (15a section 2.6 row "10, 11, 12, 13 cluster cards", index
  `ix_cluster_memberships_cluster`); 15b's cluster shapes time them at 1,000,000 (16 question C6).
- The activation key stays `uq_cluster_activations_served_key` (02d): Part A has no label scope.

### 1.2 Revision `0013_claims_cutover` additions (18b R5): what a page or a dry run showed

The stale-page rule of section 3.1 needs the server to remember what a selection dry run showed:

```sql
CREATE TABLE label_selections (               -- one dry run of a filter act ("Label all N as ..."), consumed by the act
  token        uuid PRIMARY KEY DEFAULT uuidv7(), PROJECT,
  op_kind      text NOT NULL CONSTRAINT ck_label_selections_op_kind
               CHECK (op_kind IN ('batch_label','move','batch_exclude','batch_unexclude')),
  filter_body  jsonb NOT NULL,                -- the validated item filter (src/schemas item filter model), for the audit trail
  n_items      integer NOT NULL CHECK (n_items BETWEEN 0 AND 20000),   -- MAX_BULK_SELECTION (item_selection.py:35)
  created_at   timestamptz NOT NULL DEFAULT now(),
  expires_at   timestamptz NOT NULL,          -- created_at + 1 hour; the act refuses an expired token (409 selection_expired)
  consumed_op_id uuid NULL,                   -- the label_operations row that used it; a second use is refused (409)
  UNIQUE (token, project_id)
);
CREATE INDEX ix_label_selections_expiry ON label_selections (expires_at);   -- the sweep
CREATE TABLE label_selection_items (
  token          uuid NOT NULL, PROJECT,
  item_key       bigint NOT NULL REFERENCES items(item_key) ON DELETE CASCADE,
  class_claim_id bigint NULL,                 -- item_state.class_claim_id when the dry run read it; NULL: no deciding claim
  PRIMARY KEY (token, item_key),
  FOREIGN KEY (token, project_id) REFERENCES label_selections (token, project_id) ON DELETE CASCADE
);
CREATE INDEX ix_label_selection_items_item_key ON label_selection_items (item_key);   -- FK: erasure and item deletes
-- RLS(label_selections), RLS(label_selection_items)
```

The `upload_sweep` tick (18b R3a) deletes expired selections (`DELETE ... WHERE expires_at < now()`
in batches of 1,000). Rows are written by the dry run with the 15a bulk pattern (`COPY` into an `ON
COMMIT DROP` temporary table, then `INSERT ... SELECT`, because row security refuses `COPY FROM`).

The item wire gains one field, `class_version`: `item_state.class_claim_id` rendered as a decimal
string, `null` when no claim decides the class (an intended change of R5's contract, generated into
`contracts/ts/itemWire.ts`). It is compared for equality only, never ordered.

### 1.3 Erasure (15a section 8) for Part A

| Derived thing | Rule | Where |
|---|---|---|
| `cluster_memberships` rows of the subject | explicit delete by `subject_key` (already in 15a's catalogue) | 15a |
| `partition_clusters.centroid` of a cluster that held an erased member | in the same erasure chunk, before the subject's vectors are deleted: `UPDATE partition_clusters SET centroid = (SELECT avg(v.embedding) FROM cluster_memberships m JOIN vec.<space table> v ON v.subject_key = m.subject_key WHERE m.partition_id = pc.partition_id AND m.cluster_uid = pc.cluster_uid AND m.subject_key <> ALL(:erased))`; a cluster left without members is deleted. pgvector's `avg(vector)` ([pgvector README](https://github.com/pgvector/pgvector), "Aggregates") | the trigger change below |
| the medoid key | gone with the `prototypes` column (1.1) | 1.1 |
| `label_selection_items` | cascade through `items`, indexed | 1.2 |
| `cluster_partitions.input_digest` | kept: a SHA-256 over sorted surrogate keys holds no content of the photo and names no subject; the catalogue test's column-name rule does not reach it | stands, with this reason |

02d's trigger `cluster_children_ready_immutable` refuses UPDATE while the partition is `ready`. Its
DELETE exception (`op_erasure_allows`, 15a section 8) is extended to UPDATE of `centroid` only:

```sql
-- in cluster_children_ready_immutable(): an UPDATE passes when TG_TABLE_NAME = 'partition_clusters',
-- op_erasure_allows(OLD.project_id) is true, and every column except centroid is unchanged
-- (NEW IS NOT DISTINCT FROM OLD after setting NEW.centroid := OLD.centroid in a copy)
```

`test_erasure_leaves_nothing` gains: an erased item that was the only member of a cluster leaves no
cluster; one that was a member of a larger cluster leaves a centroid equal (within 1e-6) to the mean
of the remaining members' vectors.

## 2. Part B: the schema each gated improvement adds, by its own revision

Each revision is written only when its gate in 16 section 3 passes; numbers follow the chain at
that time (`0019` and on, in the order of 16 section 3). Results tables (heads, proposals) are
rebuildable, so a later key change there is a drop and create, never a data move.

### 2.1 B1, the random check sample: no DDL

The sample is a bulk human act of kind `audit_start` writing `audit_sampled` claims (15a sections
2.2, 3.2), as today's machine-label audit does. Only the payload model changes (pydantic in
`src/domain/claim_payloads.py`, its Rust mirror in `op_domain`): `audit_sampled` gains `sample_kind:
'machine' | 'random'` and `scope` (the label scope id or `null`). The freeze of a random sample as
the test set is today's `holdout_frozen` bulk act over the sample's labelled objects.

### 2.2 B4, label scopes

```sql
CREATE TABLE label_scopes (
  scope_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, PROJECT,
  name       text NOT NULL CHECK (btrim(name) <> ''),
  is_rest    boolean NOT NULL DEFAULT false,   -- the one scope that takes every detector class no other scope names
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (scope_id, project_id), UNIQUE (project_id, name)
);
CREATE UNIQUE INDEX uq_label_scopes_rest ON label_scopes (project_id) WHERE is_rest;
CREATE TABLE label_scope_classes (            -- a detector class name belongs to at most one scope
  project_id uuid NOT NULL, scope_id bigint NOT NULL,
  detector_class_name text NOT NULL,          -- detections.class_name of the scope detector (16 section 3, B4)
  PRIMARY KEY (project_id, detector_class_name),
  FOREIGN KEY (scope_id, project_id) REFERENCES label_scopes (scope_id, project_id) ON DELETE CASCADE,
  FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX ix_label_scope_classes_scope ON label_scope_classes (scope_id);
CREATE TABLE label_scope_state (              -- one row per project: the revision of the scope definitions (OCC)
  project_id uuid PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
  revision   bigint NOT NULL DEFAULT 1, updated_at timestamptz NOT NULL DEFAULT now()
);
-- RLS(label_scopes), RLS(label_scope_classes), RLS(label_scope_state)

-- partitions and activations per scope; a NULL scope is a partition over the whole library (Part A's)
ALTER TABLE cluster_partitions
  ADD COLUMN scope_id bigint NULL,
  ADD COLUMN scope_revision bigint NULL,      -- label_scope_state.revision the pool read saw
  ADD CONSTRAINT fk_cluster_partitions_scope FOREIGN KEY (scope_id, project_id)
      REFERENCES label_scopes (scope_id, project_id) ON DELETE CASCADE,
  ADD CONSTRAINT ck_cluster_partitions_scope_revision CHECK ((scope_id IS NULL) = (scope_revision IS NULL));
CREATE INDEX ix_cluster_partitions_scope_id ON cluster_partitions (scope_id) WHERE scope_id IS NOT NULL;
ALTER TABLE cluster_activations ADD COLUMN scope_id bigint NULL,
  ADD CONSTRAINT fk_cluster_activations_scope FOREIGN KEY (scope_id, project_id)
      REFERENCES label_scopes (scope_id, project_id) ON DELETE CASCADE,
  DROP CONSTRAINT uq_cluster_activations_served_key,
  ADD CONSTRAINT uq_cluster_activations_served_key UNIQUE NULLS NOT DISTINCT
      (project_id, target, level, embedding_field, scope_id);
CREATE INDEX ix_cluster_activations_scope_id ON cluster_activations (scope_id) WHERE scope_id IS NOT NULL;
ALTER TABLE cluster_activation_events ADD COLUMN scope_id bigint NULL;   -- history, no key: a deleted scope's events stay
```

The object-to-scope mapping is not stored: it is resolved **inside the reading statement**, so the
pool read, placement and the scope list see one snapshot (review major 13):

```sql
-- the scope of every item, as one reusable view; the scope detector is the newest detector analyser
-- of the item (15a section 3.4 step 6), an item without one falls to the rest scope
CREATE VIEW item_scopes AS
SELECT s.project_id, s.item_key,
       COALESCE(c.scope_id, r.scope_id) AS scope_id
FROM item_state s
LEFT JOIN LATERAL (SELECT d.class_name FROM detections d JOIN analysers a ON a.analyser_id = d.analyser_id
                   WHERE d.item_key = s.item_key AND a.kind = 'detector'
                   ORDER BY d.analyser_id DESC LIMIT 1) d ON true
LEFT JOIN label_scope_classes c ON c.project_id = s.project_id AND c.detector_class_name = d.class_name
JOIN label_scopes r ON r.project_id = s.project_id AND r.is_rest;
-- security_invoker = true, so the caller's row security applies ([CREATE VIEW](https://www.postgresql.org/docs/18/sql-createview.html))
ALTER VIEW item_scopes SET (security_invoker = true);
```

`detections` is unique on `(item_key, analyser_id)` (15a section 2.3), which serves the lateral
read. A partition records the `scope_revision` it read; a partition whose scope revision is older
than `label_scope_state.revision` is never activated, and the tick refits every scope a scope edit
changed. An item whose detector class changes on a re-run stays a member of its old partition until
that scope's next fit, and placement uses the current mapping (section 3.4).

### 2.3 B5, the classifier on the person's labels ("the head")

One head per project and item space, over the whole library (16 section 3, B5: the evidence gives no
case for one head per scope). If M16-2 shows per-scope heads pay on real detector boxes, the head
tables are dropped and created with a scope key (they hold rebuildable results only).

```sql
CREATE TABLE head_training_state (            -- one row per project and space: the training revision (section 3.3)
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  space_id   uuid NOT NULL REFERENCES embedding_spaces(id) ON DELETE CASCADE,
  revision   bigint NOT NULL DEFAULT 1,       -- bumped in the transaction of every change to a training pair
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, space_id)
);
CREATE INDEX ix_head_training_state_space_id ON head_training_state (space_id);
CREATE TABLE head_models (
  head_id          uuid PRIMARY KEY,          -- uuidv5(job id): a retry finds and deletes its own building rows
  PROJECT,
  space_id         uuid NOT NULL REFERENCES embedding_spaces(id) ON DELETE CASCADE,
  analyser_id      integer NOT NULL REFERENCES analysers(analyser_id),   -- kind 'head', name 'class_mean', version
  training_revision bigint NOT NULL,          -- head_training_state.revision in the fit's snapshot
  n_labelled integer NOT NULL, n_classes integer NOT NULL, n_proposals integer NOT NULL,
  calibration jsonb NOT NULL,                 -- HeadCalibration model: isotonic knots, or {"state": "uncalibrated"}
  metrics     jsonb NOT NULL,                 -- HeadMetrics model: per class out-of-fold precision with count (16, B5)
  status      text NOT NULL CONSTRAINT ck_head_models_status CHECK (status IN ('building','ready','failed')),
  job_id      uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  created_at  timestamptz NOT NULL DEFAULT now(), finished_at timestamptz NULL,
  UNIQUE (head_id, project_id)
);
CREATE INDEX ix_head_models_space_id ON head_models (space_id);
CREATE INDEX ix_head_models_job_id ON head_models (job_id) WHERE job_id IS NOT NULL;
CREATE TABLE head_class_means (
  head_id uuid NOT NULL, project_id uuid NOT NULL,
  class_name text NOT NULL, class_id_hint integer NOT NULL,   -- by name with the registry hint, as a claim (15a 3.3)
  n integer NOT NULL CHECK (n > 0),
  mean vector NOT NULL,                       -- f32, as fitted
  PRIMARY KEY (head_id, class_id_hint),
  FOREIGN KEY (head_id, project_id) REFERENCES head_models (head_id, project_id) ON DELETE CASCADE
);
CREATE TABLE head_proposals (                 -- one row per unlabelled item the head scored
  head_id uuid NOT NULL, project_id uuid NOT NULL,
  item_key bigint NOT NULL REFERENCES items(item_key) ON DELETE CASCADE,
  class_name text NOT NULL, class_id_hint integer NOT NULL,
  second_class_id_hint integer NULL,
  score real NOT NULL, margin real NOT NULL,
  confidence real NULL,                       -- NULL exactly when the head is uncalibrated
  PRIMARY KEY (head_id, item_key),
  FOREIGN KEY (head_id, project_id) REFERENCES head_models (head_id, project_id) ON DELETE CASCADE
);
CREATE INDEX ix_head_proposals_item_key ON head_proposals (item_key);   -- 15a's FK rule: erasure of one photo
CREATE INDEX ix_head_proposals_page ON head_proposals
  (head_id, class_id_hint, confidence DESC NULLS LAST, margin DESC, item_key);
CREATE INDEX ix_head_proposals_margin ON head_proposals (head_id, margin, item_key);
ALTER TABLE head_proposals SET (autovacuum_vacuum_scale_factor = 0.01, autovacuum_vacuum_threshold = 1000,
                                autovacuum_vacuum_cost_limit = 2000, autovacuum_vacuum_insert_scale_factor = 0.01);
CREATE TABLE head_serving (                   -- the served head per project and space; OCC on revision
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  space_id   uuid NOT NULL REFERENCES embedding_spaces(id) ON DELETE CASCADE,
  head_id    uuid NOT NULL,
  training_revision bigint NOT NULL,          -- copied from the head; a pointer moves only to a higher value
  revision   bigint NOT NULL DEFAULT 1, updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, space_id),
  FOREIGN KEY (head_id, project_id) REFERENCES head_models (head_id, project_id) ON DELETE CASCADE
);
CREATE INDEX ix_head_serving_head_id ON head_serving (head_id);
CREATE INDEX ix_head_serving_space_id ON head_serving (space_id);
-- RLS on all six tables
ALTER TABLE analysers DROP CONSTRAINT ck_analysers_kind,
  ADD CONSTRAINT ck_analysers_kind CHECK (kind IN ('detector','open_vocab','region_stage','text_reader','vlm',
       'classifier','probe','score','projection','capture_parser','encoder','face_detector','assist',
       'raw_label_clustering','import','head'));
-- analysers is a deployment catalog of a few hundred rows: the CHECK rewrite scans nothing large (15a section 1, Names).
-- The 'head' analyser row is inserted by the first head_fit through the same get-or-create the other kinds use; no seed.

-- the training revision is bumped by one trigger, so no writer can forget it (section 3.3)
CREATE FUNCTION head_training_bump() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  UPDATE head_training_state h SET revision = revision + 1, updated_at = now()
  WHERE h.project_id IN (SELECT DISTINCT project_id FROM changed);
  RETURN NULL;
END $$;
CREATE TRIGGER item_state_head_training_upd AFTER UPDATE ON item_state
  REFERENCING OLD TABLE AS old_rows NEW TABLE AS changed FOR EACH STATEMENT
  EXECUTE FUNCTION head_training_bump();      -- the function compares old_rows and changed on
                                              -- (class_id, class_validated, excluded, holdout) and returns early when equal
CREATE TRIGGER item_state_head_training_del AFTER DELETE ON item_state
  REFERENCING OLD TABLE AS changed FOR EACH STATEMENT EXECUTE FUNCTION head_training_bump();
```

The first `head_fit` of a project and space inserts its `head_training_state` row (`INSERT ...
ON CONFLICT DO NOTHING`); before that the trigger updates nothing, which is right: no head exists
to go stale. Transition tables in statement-level triggers: [CREATE TRIGGER](https://www.postgresql.org/docs/18/sql-createtrigger.html)
("REFERENCING ... TABLE AS"). The function's early return keeps a fold write that changes no
training column from taking the row lock. Sizes at 1,000,000 unlabelled objects: a proposal row is
24 + 16 + 16 + 8 + about 20 (name) + 4 + 4 + 4 x 3 = about 104 bytes, plus three indexes, about
0.25 GB per head (arithmetic, 15a's tuple rule); retention keeps the serving head and the one before.

## 3. Correctness while people keep labelling

### 3.1 Two tabs, two people: what the person saw is the precondition (review major 11)

| Act | Precondition | On a changed object |
|---|---|---|
| Page act with explicit ids (cluster detail selection, review page, suggestion page) | the request carries, per object, the `class_version` the page showed | under the `items` lock (15a rank 6), an object whose `item_state.class_claim_id` differs from the shown value is skipped; the answer lists it as `changed_since_shown` with its current class; the dry run reports the same count |
| Filter act ("Label all N as ...", move, exclude by filter) | the dry run stores the matched objects with their `class_version` (1.2) and returns a `selection_token`; the act names the token, not the filter | the act writes only the stored objects; an object changed since the dry run is skipped and reported; objects added to the group after the dry run are not labelled (today they would be, unseen and uncounted) |
| Single label from a page | `expected_class_version` (optional for API clients; the frontend always sends it) | 409 `class_changed` with the current class; without the field, last writer wins as today |

Equality on a claim id needs no ordering: the comparison runs under the item lock, so two acts on
one object serialise (15a section 7: "a second writer waits and then folds the first one's committed
inputs"). Machine writes never change `class_claim_id` (they write opinions, 15a section 4.4), so a
VLM answer arriving after the page loaded does not block the person's act.

### 3.2 Ordering without commit-ordered ids (review major 9)

Identity and sequence values are not handed out in commit order: `nextval` values are "never
reclaimed" and concurrent sessions "receive a distinct sequence value", nothing more
([sequence functions](https://www.postgresql.org/docs/18/functions-sequence.html)); a transaction id
reflects the order of first write, not of commit
([transaction ids](https://www.postgresql.org/docs/18/transaction-id.html)). Logical decoding is
the documented commit-ordered stream ("concurrent transactions are decoded in commit order",
[output plugins](https://www.postgresql.org/docs/18/logicaldecoding-output-plugin.html)) and is more
machinery than this needs. This plan uses no id as a watermark. Every "is this result current?"
question is answered by a **counter row updated in the same transaction as the change**: concurrent
updates of one row wait for each other ("the second updater will wait for the first updating
transaction to commit or roll back", [READ COMMITTED](https://www.postgresql.org/docs/18/transaction-iso.html)),
so the counter's values are in commit order for that row, and a snapshot that sees value R sees
every change counted up to R and none after.

| Counter | Bumped by | Read by |
|---|---|---|
| `cluster_activations.revision` (02d) | activation (OCC) | activation, the tick |
| `label_scope_state.revision` (B4) | every scope edit (OCC: `revision = :expected`) | the pool read; activation refuses a partition of an older scope revision |
| `head_training_state.revision` (B5) | the trigger of 2.3, in the transaction of every label, undo, merge job chunk, holdout freeze, exclusion, import, combine, refold and erasure that changes a training column | `head_fit` (in its snapshot); the serving pointer (moves only to a higher value); the follow-up rule (3.3) |

### 3.3 The head stays fresh without a tick (review majors 8, 9, 10)

1. Any act that changes a training pair bumps `head_training_state.revision` (trigger) and, after
   commit, queues `head_fit` for the project and space. A refused insert (one already queued or
   running, `uq_jobs_one_active`, 02b) is fine because of step 3.
2. `head_fit` reads the labelled part, the unlabelled part and the revision R in one `REPEATABLE READ
   READ ONLY` transaction ([SET TRANSACTION](https://www.postgresql.org/docs/18/sql-set-transaction.html)).
3. It writes its rows, then in one transaction: moves `head_serving` to it only if `R >
   head_serving.training_revision` (else the head is deleted: a newer one already serves), and, if
   the live revision is now above R, inserts the follow-up `head_fit` in the same transaction as its
   own terminal state (the runner's finish hook, 16a section 4 for 18a R1j).
4. Freshness bound, in the person's terms: a label is in the served suggestions after at most the
   rest of the running fit plus one full fit. In labels: the labels the person gives during two fit
   durations. M16-3 measures both numbers (16 section 4).

The periodic tick queues `head_fit` only for a project whose revision is above its serving head's
and that has no active job (a crash backstop), not as the freshness path.

### 3.4 Class rename, merge, deprecate and undo, for everything a machine made (review major 10)

Class identity is the name; inside a project the registry key is the name's hint, resolved by 15a
section 3.3's one function (hint, then `merged_into` to the end, then the active class of the
`name_key`, else unresolved; `test_class_resolution` covers rename and merge).

| Thing that names a class | Rename | Merge (source into target) | Deprecate | Undo |
|---|---|---|---|---|
| Claims (`class_set`) | resolved at fold by the function | the `class_merge` job moves `item_state.class_id` in chunks (15a section 7); claims untouched | refused while any item carries the class (409 `class_still_referenced`, today's rule) | a withdraw; the fold restores |
| Machine opinions (`class_opinions`: VLM, classifier, propagation, import suggestion; `probe_predictions`) | resolved at fold or read by the same function | resolve to the target through `merged_into` | an opinion naming a deprecated, unmerged class resolves to nothing: no machine class shown (fail closed) | an adoption's or propagation's op undo deletes its rows (15a 3.5) |
| Cluster cards (dominant class, labelled counts) | joined at read from `item_state.class_id` | the merge root (readers group by the root, 15a section 7) | n/a | n/a |
| Assist answers (`assist_answers.class_name`), audit verdicts (`audit_sampled` payload `label_name`) | read by name through the function; an audit verdict keeps the name it judged, by design (it is history) | as rename | as opinions | n/a |
| Head proposals and class means (B5) | resolved at read by the function from `(class_name, class_id_hint)`; pages filter `class_id_hint = ANY(:family)` (the class and every class merged into it), as 15a's readers do | resolved to the target; the merge job's chunks change training pairs, so the trigger bumps the revision and a refit follows | resolves to nothing: the proposal is not shown, the revision bump refits | label undo: a training change, so a refit follows; there is no merge undo today (a merged class cannot be restored, `classes_write.py`), and none is added |
| Trained models | class names in the model's own list, by name across the boundary (VISION_AND_GOALS) | a model trained before a merge keeps the old names; promotion maps by name | as merge | n/a |

`cluster_identities.promoted_class_id` and `name` have no writer today (survey) and get none here.

### 3.5 Erasure of what the loop derives (review major 15)

Part A's rows are in 1.3. Part B adds, in the erasure chunk that deletes the subject's items:

| Derived thing | Rule |
|---|---|
| `head_proposals` of the item | cascade through `items`, indexed (`ix_head_proposals_item_key`) |
| Heads whose training set or proposals held the item (their means and calibration are computed from its vector) | `DELETE FROM head_models WHERE project_id = :p AND space_id = ANY(:spaces of the erased vectors)`: cascades to means, proposals and the serving pointer; the trigger's revision bump on the item's `item_state` delete makes the after-commit `head_fit` refit. The person sees "suggestions refreshing" until it serves |
| `head_models.metrics` per class | deleted with the head |
| Label scopes | hold detector class names only, no subject |

Catalogue: `head_proposals` (cascade), `head_models` (explicit delete per space), and
`label_selection_items` (cascade) join 15a section 8's table; `test_erasure_catalogue_complete`
finds `item_key` columns by name and so covers them.

### 3.6 Jobs that crash, jobs that race

| Case | Answer |
|---|---|
| A `cluster_run` dies | its partition stays `building`; a retry with the same job id first deletes its own `building` partition (cascade), then fits again; readers see only `ready` partitions through the pointer (02d) |
| A `head_fit` dies | `head_id` = uuidv5 of the job id; the retry deletes that head's rows first |
| Two `cluster_place` of one target and field | refused by the exclusive group; the running one loops until no unplaced object remains, the tick is the backstop for an insert refused at its last moment |
| A refit during a person's session | the page cursor carries the partition id; the person finishes the group; the next list shows the new partition; labels are claims on objects, independent of groups |
| Many API processes and workers | all shared state is in PostgreSQL (pointers with OCC, counters, jobs with leases and fencing tokens); in-process data is a cache keyed by an immutable id (a partition's centroids) or a job's working copy |

### 3.7 Lock order (15a section 7) additions

| Rank | Row | Taken by |
|---|---|---|
| after (6) `items` | `head_training_state` (one row per project and space; B5) | the trigger, inside any writer that already holds its items; never before them |
| (1) | `label_selections` row `FOR UPDATE` (Part A) | the filter act: the selection row first, then its new operation row (both rank 1, in that order), then (6) items |
| none | `label_scope_state` (B4) | scope edits only (configuration; no item lock) |

Placement and cluster runs take no item lock: they write memberships, which no fold reads.
