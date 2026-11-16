"""Catalogs, photos, item identity, entities and the claim log of revision 0010 (15a sections 2.1-2.3)."""

from __future__ import annotations

from src.db.migrations.schema_0010.ddl import IDENTITY, PROJECT, Table, ck, closed_set, fk, pk, uq


ANALYSER_KINDS = (
    'detector', 'open_vocab', 'region_stage', 'text_reader', 'vlm', 'classifier', 'probe',
    'score', 'projection', 'capture_parser', 'encoder', 'face_detector', 'assist',
    'raw_label_clustering', 'import',
)  # fmt: skip
SUBJECT_KINDS = ('photo', 'media', 'item', 'box', 'pair', 'face', 'entity', 'claim')
ACTOR_KINDS = ('human', 'import', 'combine')
ENTITY_KINDS = ('person', 'place', 'event', 'trip', 'album', 'keyword', 'memory')

ANALYSERS = Table(
    'analysers',
    (
        f'analyser_id integer {IDENTITY}',
        'kind text NOT NULL',
        'name text NOT NULL',
        'version text NOT NULL',
        'config_sha256 text NOT NULL',
        "config_ref jsonb NOT NULL DEFAULT '{}'",
        'created_at timestamptz NOT NULL DEFAULT now()',
    ),
    (
        pk('analyser_id'),
        uq('uq_analysers_identity', 'kind', 'name', 'version', 'config_sha256'),
        ck('kind', closed_set('kind', ANALYSER_KINDS)),
    ),
)

CLAIM_KINDS = Table(
    'claim_kinds',
    (
        'kind text NOT NULL',
        'subject_kind text NOT NULL',
        'layer text NOT NULL',
        'pins_geometry boolean NOT NULL',
        'asserts_object boolean NOT NULL',
    ),
    (
        pk('kind'),
        uq('uq_claim_kinds_kind_subject_kind', 'kind', 'subject_kind'),
        ck('subject_kind', closed_set('subject_kind', SUBJECT_KINDS)),
    ),
)

PHOTOS = Table(
    'photos',
    (
        f'photo_key bigint {IDENTITY}',
        PROJECT,
        'image_id uuid NOT NULL',
        'imohash bytea NOT NULL',
        'content_sha256 bytea NULL',
        'storage_key text NOT NULL',
        'size_bytes bigint NOT NULL',
        'width integer NOT NULL',
        'height integer NOT NULL',
        'frame smallint NOT NULL',
        "source_state text NOT NULL DEFAULT 'present'",
        'decode_route text NOT NULL',
        'source text NOT NULL',
        'source_identifier text NULL',
        'ingest_run_tag text NULL',
        'ingest_job_id uuid NULL -> jobs(id) SET NULL',
        'ingest_request_id text NULL',
        'client_metadata jsonb NULL',
        'created_at timestamptz NOT NULL DEFAULT now()',
    ),
    (
        pk('photo_key'),
        uq('uq_photos_image_id', 'project_id', 'image_id'),
        ck('imohash', 'length(imohash) = 16'),
        ck('content_sha256', 'content_sha256 IS NULL OR length(content_sha256) = 32'),
        ck('size', 'size_bytes >= 0'),
        ck('width', 'width > 0'),
        ck('height', 'height > 0'),
        ck('frame', 'frame BETWEEN 1 AND 8'),
        ck('source_state', "source_state IN ('present', 'moved', 'missing')"),
    ),
)

PHOTO_PROVENANCE = Table(
    'photo_provenance',
    (
        f'provenance_key bigint {IDENTITY}',
        'photo_key bigint NOT NULL -> photos(photo_key) CASCADE',
        PROJECT,
        'kind text NOT NULL',
        'import_id uuid NULL -> dataset_imports(id) CASCADE',
        'combine_job_id uuid NULL -> combine_runs(job_id) SET NULL',
        'dataset_split text NULL',
        'source_stem text NULL',
        'stratum text NULL',
        'hard_negative boolean NOT NULL DEFAULT false',
        'label_state text NULL',
        "negative_for text[] NOT NULL DEFAULT '{}'",
        'origin_project_slug text NULL',
        'origin_image_id uuid NULL',
        'created_at timestamptz NOT NULL DEFAULT now()',
    ),
    (
        pk('provenance_key'),
        ck('kind', "kind IN ('import', 'combine')"),
        ck(
            'label_state',
            "label_state IS NULL OR label_state IN ('labeled', 'negative', 'unlabeled')",
        ),
        ck('import_kind', "(kind = 'import') = (import_id IS NOT NULL)"),
    ),
)

ITEMS = Table(
    'items',
    (
        f'item_key bigint {IDENTITY}',
        PROJECT,
        'photo_key bigint NOT NULL -> photos(photo_key) CASCADE',
        'crop_id uuid NOT NULL',
        'box_seq integer NOT NULL DEFAULT 0',
        'geometry_seq integer NOT NULL DEFAULT 0',
        'region_seq integer NOT NULL DEFAULT 0',
        'created_at timestamptz NOT NULL DEFAULT now()',
    ),
    (pk('item_key'), uq('uq_items_crop_id', 'project_id', 'crop_id')),
)

ENTITIES = Table(
    'entities',
    (
        f'entity_key bigint {IDENTITY}',
        PROJECT,
        'entity_id uuid NOT NULL DEFAULT uuidv7()',
        'kind text NOT NULL',
        'analyser_id integer NULL -> analysers(analyser_id)',
        'cluster_uid uuid NULL -> cluster_identities(cluster_uid) SET NULL',
        'geonames_id integer NULL',
        'created_at timestamptz NOT NULL DEFAULT now()',
    ),
    (
        pk('entity_key'),
        uq('uq_entities_entity_id', 'project_id', 'entity_id'),
        ck('kind', closed_set('kind', ENTITY_KINDS)),
    ),
)

CLAIMS = Table(
    'claims',
    (
        f'claim_id bigint {IDENTITY}',
        PROJECT,
        'at timestamptz NOT NULL DEFAULT now()',
        'actor_kind text NOT NULL',
        'actor text NOT NULL',
        'actor_user_id uuid NULL',
        'op_id uuid NULL',
        'kind text NOT NULL',
        'subject_kind text NOT NULL',
        'photo_key bigint NULL -> photos(photo_key) CASCADE',
        'subject_crop_id uuid NULL',
        'subject_box_id text NULL',
        'subject_face_id uuid NULL',
        'subject_pair_id uuid NULL',
        'subject_video_key bigint NULL',
        'subject_entity_key bigint NULL -> entities(entity_key) CASCADE',
        'subject_bbox real[] NULL',
        'subject_machine_bbox real[] NULL',
        'subject_frame smallint NULL',
        'subject_key_hint bigint NULL',
        # CASCADE (deviation 2 of the lane report): a withdraw dies with its target, so a batched
        # project delete or an erasure never meets a withdraw whose target is already gone
        'withdraws_claim_id bigint NULL -> claims(claim_id) CASCADE',
        "payload jsonb NOT NULL DEFAULT '{}'",
    ),
    (
        pk('claim_id'),
        fk(('kind', 'subject_kind'), 'claim_kinds', ('kind', 'subject_kind')),
        ck('actor_kind', closed_set('actor_kind', ACTOR_KINDS)),
        ck('subject_bbox', 'subject_bbox IS NULL OR cardinality(subject_bbox) = 4'),
        ck(
            'subject_machine_bbox',
            'subject_machine_bbox IS NULL OR cardinality(subject_machine_bbox) = 4',
        ),
        ck('withdraw_target', "(kind = 'withdraw') = (withdraws_claim_id IS NOT NULL)"),
        ck(
            'item_subject',
            "subject_kind NOT IN ('item', 'box') OR (photo_key IS NOT NULL AND "
            'subject_crop_id IS NOT NULL AND subject_bbox IS NOT NULL AND subject_frame IS NOT NULL)',
        ),
        ck('box_subject', "subject_kind <> 'box' OR subject_box_id IS NOT NULL"),
        ck(
            'media_subject',
            "subject_kind <> 'media' OR (photo_key IS NULL) <> (subject_video_key IS NULL)",
        ),
    ),
)

TABLES = (ANALYSERS, CLAIM_KINDS, PHOTOS, PHOTO_PROVENANCE, ITEMS, ENTITIES, CLAIMS)

INDEXES = (
    (
        'CREATE UNIQUE INDEX uq_photos_sha256 ON photos (project_id, content_sha256) '
        'WHERE content_sha256 IS NOT NULL'
    ),
    (
        'CREATE UNIQUE INDEX uq_photo_provenance_source ON photo_provenance '
        '(photo_key, kind, import_id, origin_project_slug, origin_image_id) NULLS NOT DISTINCT'
    ),
    'CREATE INDEX ix_claims_photo ON claims (photo_key, claim_id) WHERE photo_key IS NOT NULL',
    (
        'CREATE INDEX ix_claims_entity ON claims (subject_entity_key, claim_id) '
        'WHERE subject_entity_key IS NOT NULL'
    ),
    (
        'CREATE UNIQUE INDEX uq_claims_withdraws ON claims (withdraws_claim_id) '
        'WHERE withdraws_claim_id IS NOT NULL'
    ),
    (
        'CREATE UNIQUE INDEX uq_claims_op_subject ON claims (op_id, kind, photo_key, subject_crop_id, '
        'subject_box_id, subject_face_id, subject_pair_id, subject_video_key, subject_entity_key) '
        "NULLS NOT DISTINCT WHERE op_id IS NOT NULL AND kind <> 'withdraw'"
    ),
)
