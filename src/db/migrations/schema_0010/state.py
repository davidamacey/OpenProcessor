"""The fold's output tables, the gallery tables, vector gaps and erasure jobs of revision 0010
(15a sections 2.4, 2.5 and 8)."""

from __future__ import annotations

from src.db.migrations.schema_0010.ddl import IDENTITY, PROJECT, Table, ck, closed_set, pk, uq


_ANALYSER = 'analyser_id integer NOT NULL -> analysers(analyser_id)'
_PHOTO = 'photo_key bigint NOT NULL -> photos(photo_key) CASCADE'
_ENTITY = 'entity_key bigint NOT NULL -> entities(entity_key) CASCADE'
GEOMETRY_SOURCES = ('detector', 'human', 'import')
# 15a section 2.4: a no-op fold writes no tuple and a label write is HOT; autovacuum starts at 1 %
AUTOVACUUM = (
    'ALTER TABLE {table} SET (autovacuum_vacuum_scale_factor = 0.01, '
    'autovacuum_vacuum_threshold = 1000, autovacuum_vacuum_cost_limit = 2000)'
)

ITEM_STATE = Table(
    'item_state',
    (
        'item_key bigint NOT NULL -> items(item_key) CASCADE',
        PROJECT,
        'photo_key bigint NOT NULL',
        'crop_id uuid NOT NULL',
        'bbox real[] NOT NULL',
        'area real NOT NULL',
        'rank smallint NOT NULL',
        'geometry_source text NOT NULL',
        'geometry_revision integer NOT NULL',
        'class_id integer NULL',
        'class_source text NULL',
        'label_source text NULL',
        'class_validated boolean NOT NULL DEFAULT false',
        'class_unresolved boolean NOT NULL DEFAULT false',
        'class_labeled_at timestamptz NULL',
        'class_confidence real NULL',
        'class_claim_id bigint NULL',
        'class_op_id uuid NULL',
        'class_opinion_source text NULL',
        'vlm_suggestion_dismissed boolean NOT NULL DEFAULT false',
        'needs_new_class boolean NOT NULL DEFAULT false',
        'probe_dismissed_class text NULL',
        'excluded boolean NOT NULL DEFAULT false',
        'excluded_reason text NULL',
        'excluded_at timestamptz NULL',
        'holdout boolean NOT NULL DEFAULT false',
        'review_dismissed_at timestamptz NULL',
        'audit_sample boolean NOT NULL DEFAULT false',
        'audit_outcome text NULL',
        'combine_conflict boolean NOT NULL DEFAULT false',
        'import_ids uuid[] NULL',
        'region_status text NULL',
        'region_validated boolean NOT NULL DEFAULT false',
        'region_verifier text NULL',
        'region_label_source text NULL',
        'region_count smallint NOT NULL DEFAULT 0',
        'region_rejected_count smallint NOT NULL DEFAULT 0',
        'region_max_score real NULL',
        'region_revision integer NOT NULL',
        'region_set_complete boolean NULL',
        'region_seen_seq integer NOT NULL DEFAULT 0',
        'geometry_pinned boolean NOT NULL DEFAULT false',
        'moved_partition_id uuid NULL',
        'moved_cluster_uid uuid NULL',
        'created_at timestamptz NOT NULL',
        'updated_at timestamptz NOT NULL',
    ),
    (pk('item_key'), ck('geometry_source', closed_set('geometry_source', GEOMETRY_SOURCES))),
)

BOX_STATE = Table(
    'box_state',
    (
        f'box_row_key bigint {IDENTITY}',
        PROJECT,
        'item_key bigint NOT NULL -> items(item_key) CASCADE',
        'box_id text NOT NULL',
        'box_key bigint NULL -> region_boxes(box_key) SET NULL',
        'bbox real[] NOT NULL',
        'state text NOT NULL',
        'score real NULL',
        'detector text NULL',
        'detector_version text NULL',
        'source text NULL',
        'bbox_correct boolean NULL',
        'confidence text NULL',
        'rejection_reason text NULL',
        'text text NULL',
        'text_source text NULL',
        'text_choice text NULL',
        'text_disagreement boolean NULL',
        'text_fold text NULL',
        'ordinal smallint NOT NULL',
        'detected_at timestamptz NULL',
    ),
    (pk('box_row_key'), uq('uq_box_state_item_box', 'item_key', 'box_id')),
)

ENTITY_EDGES = Table(
    'entity_edges',
    (
        _ENTITY,
        PROJECT,
        'subject_kind text NOT NULL',
        'subject_key bigint NOT NULL',
        _ANALYSER,
        'weight real NULL',
    ),
    (
        pk('entity_key', 'subject_kind', 'subject_key'),
        ck('subject_kind', "subject_kind IN ('photo', 'face', 'item')"),
    ),
)

ENTITY_STATE = Table(
    'entity_state',
    (
        _ENTITY,
        PROJECT,
        'kind text NOT NULL',
        'name text NULL',
        'name_key text NULL',
        'parent_key bigint NULL -> entities(entity_key) CASCADE',
        'path text NULL',
        'description text NULL',
        'cover_photo_key bigint NULL -> photos(photo_key) SET NULL',
        'sort text NULL',
        'hidden boolean NOT NULL DEFAULT false',
        'merged_into bigint NULL',
        'state text NOT NULL',
        'member_count integer NOT NULL DEFAULT 0',
        'revision bigint NOT NULL',
        'updated_at timestamptz NOT NULL',
    ),
    (
        pk('entity_key'),
        ck('sort', "sort IS NULL OR sort IN ('position', 'taken_asc', 'taken_desc')"),
    ),
)

MEMBERSHIP_STATE = Table(
    'membership_state',
    (
        _ENTITY,
        PROJECT,
        'subject_kind text NOT NULL',
        'subject_key bigint NOT NULL',
        'source text NOT NULL',
        'position numeric NULL',
        'weight real NULL',
    ),
    (
        pk('entity_key', 'subject_kind', 'subject_key'),
        ck('source', "source IN ('machine', 'human')"),
    ),
)

FACE_STATE = Table(
    'face_state',
    (
        'face_key bigint NOT NULL -> faces(face_key) CASCADE',
        PROJECT,
        'person_entity_key bigint NULL -> entities(entity_key) SET NULL',
        'person_state text NOT NULL',
        'assign_score real NULL',
        'assign_source text NULL',
    ),
    (
        pk('face_key'),
        ck('person_state', "person_state IN ('unassigned', 'suggested', 'assigned', 'confirmed')"),
    ),
)

PHOTO_STATE = Table(
    'photo_state',
    (
        _PHOTO,
        PROJECT,
        'taken_local timestamp NULL',
        'taken_utc timestamp NULL',
        'time_source text NULL',
        'lat double precision NULL',
        'lon double precision NULL',
        'location_source text NULL',
        'favourite boolean NOT NULL DEFAULT false',
        "pick text NOT NULL DEFAULT 'none'",
        'hidden boolean NOT NULL DEFAULT false',
        'archived boolean NOT NULL DEFAULT false',
        'trashed_at timestamptz NULL',
        'title text NULL',
        'caption text NULL',
        'stack_pick_group uuid NULL',
        'revision bigint NOT NULL',
        'updated_at timestamptz NOT NULL',
    ),
    (pk('photo_key'),),
)

PHOTO_FILES = Table(
    'photo_files',
    (
        _PHOTO,
        PROJECT,
        'role text NOT NULL',
        'storage_key text NOT NULL',
        'size_bytes bigint NOT NULL',
        'content_sha256 bytea NULL',
        "source_state text NOT NULL DEFAULT 'present'",
    ),
    (
        pk('photo_key', 'role', 'storage_key'),
        ck('role', "role IN ('motion', 'edited', 'recipe', 'raw', 'sidecar')"),
        ck('source_state', "source_state IN ('present', 'moved', 'missing')"),
    ),
)

PHOTO_DERIVATIVES = Table(
    'photo_derivatives',
    (
        _PHOTO,
        PROJECT,
        _ANALYSER,
        'thumbhash bytea NULL',
        'aspect real NOT NULL',
        'grid_key text NULL',
        'at timestamptz NOT NULL DEFAULT now()',
    ),
    (pk('photo_key'),),
)

PHOTO_SIGNALS = Table(
    'photo_signals',
    (
        _PHOTO,
        PROJECT,
        _ANALYSER,
        'sharpness real NULL',
        'exposure real NULL',
        'aesthetic real NULL',
        'utility_class text NULL',
    ),
    (pk('photo_key'),),
)

PHOTO_PLACES = Table(
    'photo_places',
    (
        _PHOTO,
        PROJECT,
        _ANALYSER,
        'geonames_id integer NULL',
        'country_code text NULL',
        'admin1 text NULL',
        'admin2 text NULL',
        'locality text NULL',
    ),
    (pk('photo_key'),),
)

VECTOR_GAPS = Table(
    'vector_gaps',
    (
        'space_id uuid NOT NULL -> embedding_spaces(id) CASCADE',
        PROJECT,
        'subject_key bigint NOT NULL',
        'reason text NOT NULL',
    ),
    (pk('space_id', 'subject_key'), ck('reason', "reason IN ('not_selected', 'failed')")),
)

ERASURE_JOBS = Table(
    'erasure_jobs',
    (
        'id uuid NOT NULL DEFAULT uuidv7()',
        PROJECT,
        'subject_kind text NOT NULL',
        'photo_keys bigint[] NULL',
        'entity_key bigint NULL',
        'state text NOT NULL',
        "counts jsonb NOT NULL DEFAULT '{}'",
        'job_id uuid NULL -> jobs(id) SET NULL',
        'created_at timestamptz NOT NULL DEFAULT now()',
        'finished_at timestamptz NULL',
    ),
    (
        pk('id'),
        ck('subject_kind', "subject_kind IN ('photo', 'person', 'project')"),
        ck('state', closed_set('state', ('running', 'done', 'failed'))),
    ),
)

TABLES = (
    ITEM_STATE, BOX_STATE, ENTITY_EDGES, ENTITY_STATE, MEMBERSHIP_STATE, FACE_STATE, PHOTO_STATE,
    PHOTO_FILES, PHOTO_DERIVATIVES, PHOTO_SIGNALS, PHOTO_PLACES, VECTOR_GAPS, ERASURE_JOBS,
)  # fmt: skip

INDEXES = (
    'CREATE INDEX ix_item_state_class_op ON item_state (class_op_id) WHERE class_op_id IS NOT NULL',
    (
        'CREATE UNIQUE INDEX uq_entity_state_keyword ON entity_state (project_id, parent_key, name_key) '
        "NULLS NOT DISTINCT WHERE kind = 'keyword'"
    ),
)
