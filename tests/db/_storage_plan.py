"""The objects of docs/design/v070_rebuild/15a_storage_schema.md sections 2 and 8, written by hand.

Written out from the plan, not read from the models or the migration, so a model and a migration
that drift together still fail. The offline tests compare the models and the rendered migration
with these sets; the database tests compare the migrated database.
"""

from __future__ import annotations


def _columns(text: str) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for line in text.strip().splitlines():
        table, _, names = line.partition(':')
        out[table.strip()] = tuple(names.split())
    return out


# table: columns in plan order (15a sections 2.1 to 2.5 and 8)
COLUMNS = _columns(
    """
analysers: analyser_id kind name version config_sha256 config_ref created_at
claim_kinds: kind subject_kind layer pins_geometry asserts_object
photos: photo_key project_id image_id imohash content_sha256 storage_key size_bytes width height frame source_state decode_route source source_identifier ingest_run_tag ingest_job_id ingest_request_id client_metadata created_at
photo_provenance: provenance_key photo_key project_id kind import_id combine_job_id dataset_split source_stem stratum hard_negative label_state negative_for origin_project_slug origin_image_id created_at
items: item_key project_id photo_key crop_id box_seq geometry_seq region_seq created_at
entities: entity_key project_id entity_id kind analyser_id cluster_uid geonames_id created_at
claims: claim_id project_id at actor_kind actor actor_user_id op_id kind subject_kind photo_key subject_crop_id subject_box_id subject_face_id subject_pair_id subject_video_key subject_entity_key subject_bbox subject_machine_bbox subject_frame subject_key_hint withdraws_claim_id payload
analysis_ledger: photo_key project_id kind analyser_id state reason updated_at
photo_capture: photo_key project_id analyser_id taken_local taken_utc offset_minutes time_source offset_source time_precision camera_make camera_model lens_model focal_length_mm focal_length_35mm orientation exif_status has_location lat lon altitude_m location_accuracy_m heading_deg location_precision burst_id
detections: detection_key project_id photo_key item_key analyser_id bbox class_name class_id confidence proposal_name blur_lap_var blur_lap_ratio blur_full_var prompt mask_polygon import_id created_at
class_opinions: item_key project_id source analyser_id class_name class_id_hint outcome confidence confidence_cat raw_label raw_label_conf raw_class proposed_class empty_reason via op_id import_id created_at
raw_label_clusters: project_id raw_label analyser_id cluster_id cluster_name distance
item_region_results: item_key project_id analyser_id status reason verified verified_at verifier verifier_version auto_confirmed visible skip_verify gate_skip detector_chain set_complete detected_at
region_boxes: box_key project_id item_key box_id analyser_id region_seq bbox state score detector detector_version source confidence rejection_reason detected_at import_id
box_texts: box_key project_id reader analyser_id text text_raw text_frame invalid confidence conf_char_min jurisdiction jurisdiction_conf
text_lines: line_key project_id photo_key item_key analyser_id seq text text_raw box_norm polygon det_score rec_score rel_height language
probe_predictions: item_key project_id analyser_id class_name class_id_hint entropy confidence margin scored_at
item_scores: item_key project_id kind analyser_id score scored_at
item_projections: item_key project_id analyser_id x y
assist_answers: item_key project_id analyser_id run_id run_created_at class_name class_confidence attributes box_attributes
item_notes: note_key project_id item_key kind analyser_id import_id body created_at
faces: face_key project_id photo_key face_id analyser_id person_item_key bbox landmarks det_score quality yaw_deg eyes_open side_px
vector_gaps: space_id project_id subject_key reason
item_state: item_key project_id photo_key crop_id bbox area rank geometry_source geometry_revision class_id class_source label_source class_validated class_unresolved class_labeled_at class_confidence class_claim_id class_op_id class_opinion_source vlm_suggestion_dismissed needs_new_class probe_dismissed_class excluded excluded_reason excluded_at holdout review_dismissed_at audit_sample audit_outcome combine_conflict import_ids region_status region_validated region_verifier region_label_source region_count region_rejected_count region_max_score region_revision region_set_complete region_seen_seq geometry_pinned moved_partition_id moved_cluster_uid created_at updated_at
box_state: box_row_key project_id item_key box_id box_key bbox state score detector detector_version source bbox_correct confidence rejection_reason text text_source text_choice text_disagreement text_fold ordinal detected_at
entity_edges: entity_key project_id subject_kind subject_key analyser_id weight
entity_state: entity_key project_id kind name name_key parent_key path description cover_photo_key sort hidden merged_into state member_count revision updated_at
membership_state: entity_key project_id subject_kind subject_key source position weight
face_state: face_key project_id person_entity_key person_state assign_score assign_source
photo_state: photo_key project_id taken_local taken_utc time_source lat lon location_source favourite pick hidden archived trashed_at title caption stack_pick_group revision updated_at
photo_files: photo_key project_id role storage_key size_bytes content_sha256 source_state
photo_derivatives: photo_key project_id analyser_id thumbhash aspect grid_key at
photo_signals: photo_key project_id analyser_id sharpness exposure aesthetic utility_class
photo_places: photo_key project_id analyser_id geonames_id country_code admin1 admin2 locality
erasure_jobs: id project_id subject_kind photo_keys entity_key state counts job_id created_at finished_at
"""
)  # fmt: skip

TABLES = set(COLUMNS)

# Deployment catalogs: no project column, no row-level security.
UNPROTECTED = {'analysers', 'claim_kinds'}

# The gallery tables of 15a section 2.5: created now, filled in phase 5.
GALLERY_TABLES = {
    'entities', 'entity_edges', 'entity_state', 'membership_state', 'face_state', 'photo_state',
    'photo_files', 'photo_derivatives', 'photo_signals', 'photo_places',
}  # fmt: skip

POLICIES = {(t, f'p_{t}_project') for t in TABLES - UNPROTECTED}

FUNCTIONS = {'claims_append_only', 'analysers_immutable', 'op_erasure_allows'}

TRIGGERS = {('claims', 'claims_append_only'), ('analysers', 'analysers_immutable')}

# Named unique constraints (not indexes).
UNIQUE_CONSTRAINTS = {
    'uq_analysers_identity', 'uq_claim_kinds_kind_subject_kind', 'uq_photos_image_id',
    'uq_items_crop_id', 'uq_entities_entity_id', 'uq_detections_item_analyser',
    'uq_region_boxes_item_box', 'uq_faces_face_id', 'uq_box_state_item_box',
}  # fmt: skip

# Unique and partial indexes the plan names (15a sections 2.1 to 2.4).
PARTIAL_INDEXES = {
    'uq_photos_sha256', 'ix_claims_photo', 'ix_claims_entity', 'uq_claims_withdraws',
    'uq_claims_op_subject', 'ix_item_state_class_op', 'uq_entity_state_keyword',
}  # fmt: skip

NULLS_NOT_DISTINCT = {
    'uq_photo_provenance_source',
    'uq_claims_op_subject',
    'uq_entity_state_keyword',
}

PLAIN_INDEXES = {'uq_photo_provenance_source'}

# Every foreign key whose referenced row can be deleted gets an index on its referencing column
# (15a section 1, "Indexes"), unless a key already leads with it. analyser_id and the claim-kind
# keys reference rows that are never deleted. (table, column, partial)
FOREIGN_KEY_INDEXES = (
    {(t, 'project_id', False) for t in (
        'photo_provenance', 'claims', 'analysis_ledger', 'photo_capture', 'detections',
        'class_opinions', 'item_region_results', 'region_boxes', 'box_texts', 'text_lines',
        'probe_predictions', 'item_scores', 'item_projections', 'assist_answers', 'item_notes',
        'vector_gaps', 'item_state', 'box_state', 'entity_edges', 'entity_state',
        'membership_state', 'face_state', 'photo_state', 'photo_files', 'photo_derivatives',
        'photo_signals', 'photo_places', 'erasure_jobs',
    )}
    | {
        ('photos', 'ingest_job_id', True), ('photo_provenance', 'import_id', True),
        ('photo_provenance', 'combine_job_id', True), ('items', 'photo_key', False),
        ('entities', 'cluster_uid', True), ('detections', 'photo_key', False),
        ('detections', 'import_id', True), ('class_opinions', 'import_id', True),
        ('region_boxes', 'import_id', True), ('text_lines', 'photo_key', False),
        ('text_lines', 'item_key', True), ('item_notes', 'item_key', False),
        ('item_notes', 'import_id', True), ('faces', 'photo_key', False),
        ('faces', 'person_item_key', True), ('box_state', 'box_key', True),
        ('entity_state', 'parent_key', True), ('entity_state', 'cover_photo_key', True),
        ('face_state', 'person_entity_key', True), ('erasure_jobs', 'job_id', True),
    }
)  # fmt: skip

FOREIGN_KEY_INDEX_NAMES = {f'ix_{t}_{c}' for t, c, _ in FOREIGN_KEY_INDEXES}

# (table, column, referenced table, ON DELETE) as PostgreSQL stores it: c cascade, n set null,
# a no action.
FOREIGN_KEYS = (
    {(t, 'project_id', 'projects', 'c') for t in TABLES - UNPROTECTED}
    | {(t, 'analyser_id', 'analysers', 'a') for t in (
        'entities', 'analysis_ledger', 'photo_capture', 'detections', 'class_opinions',
        'raw_label_clusters', 'item_region_results', 'region_boxes', 'box_texts', 'text_lines',
        'probe_predictions', 'item_scores', 'item_projections', 'assist_answers', 'item_notes',
        'faces', 'entity_edges', 'photo_derivatives', 'photo_signals', 'photo_places',
    )}
    | {
        ('photos', 'ingest_job_id', 'jobs', 'n'),
        ('photo_provenance', 'photo_key', 'photos', 'c'),
        ('photo_provenance', 'import_id', 'dataset_imports', 'c'),
        ('photo_provenance', 'combine_job_id', 'combine_runs', 'n'),
        ('items', 'photo_key', 'photos', 'c'),
        ('entities', 'cluster_uid', 'cluster_identities', 'n'),
        ('claims', 'photo_key', 'photos', 'c'),
        ('claims', 'subject_entity_key', 'entities', 'c'),
        ('claims', 'withdraws_claim_id', 'claims', 'c'),
        ('claims', 'kind', 'claim_kinds', 'a'),
        ('analysis_ledger', 'photo_key', 'photos', 'c'),
        ('photo_capture', 'photo_key', 'photos', 'c'),
        ('detections', 'photo_key', 'photos', 'c'),
        ('detections', 'item_key', 'items', 'c'),
        ('detections', 'import_id', 'dataset_imports', 'c'),
        ('class_opinions', 'item_key', 'items', 'c'),
        ('class_opinions', 'import_id', 'dataset_imports', 'c'),
        ('item_region_results', 'item_key', 'items', 'c'),
        ('region_boxes', 'item_key', 'items', 'c'),
        ('region_boxes', 'import_id', 'dataset_imports', 'c'),
        ('box_texts', 'box_key', 'region_boxes', 'c'),
        ('text_lines', 'photo_key', 'photos', 'c'),
        ('text_lines', 'item_key', 'items', 'c'),
        ('probe_predictions', 'item_key', 'items', 'c'),
        ('item_scores', 'item_key', 'items', 'c'),
        ('item_projections', 'item_key', 'items', 'c'),
        ('assist_answers', 'item_key', 'items', 'c'),
        ('item_notes', 'item_key', 'items', 'c'),
        ('item_notes', 'import_id', 'dataset_imports', 'c'),
        ('faces', 'photo_key', 'photos', 'c'),
        ('faces', 'person_item_key', 'items', 'n'),
        ('vector_gaps', 'space_id', 'embedding_spaces', 'c'),
        ('item_state', 'item_key', 'items', 'c'),
        ('box_state', 'item_key', 'items', 'c'),
        ('box_state', 'box_key', 'region_boxes', 'n'),
        ('entity_edges', 'entity_key', 'entities', 'c'),
        ('entity_state', 'entity_key', 'entities', 'c'),
        ('entity_state', 'parent_key', 'entities', 'c'),
        ('entity_state', 'cover_photo_key', 'photos', 'n'),
        ('membership_state', 'entity_key', 'entities', 'c'),
        ('face_state', 'face_key', 'faces', 'c'),
        ('face_state', 'person_entity_key', 'entities', 'n'),
        ('photo_state', 'photo_key', 'photos', 'c'),
        ('photo_files', 'photo_key', 'photos', 'c'),
        ('photo_derivatives', 'photo_key', 'photos', 'c'),
        ('photo_signals', 'photo_key', 'photos', 'c'),
        ('photo_places', 'photo_key', 'photos', 'c'),
        ('erasure_jobs', 'job_id', 'jobs', 'n'),
    }
)  # fmt: skip

# Tables whose primary key is a bigint identity column ALWAYS (15a section 1, "Keys inside").
IDENTITY_KEYS = {
    'photos': 'photo_key', 'photo_provenance': 'provenance_key', 'items': 'item_key',
    'entities': 'entity_key', 'claims': 'claim_id', 'detections': 'detection_key',
    'region_boxes': 'box_key', 'text_lines': 'line_key', 'item_notes': 'note_key',
    'faces': 'face_key', 'box_state': 'box_row_key',
}  # fmt: skip

# Tables a subject key lives in: a row of each must be reachable by erasure (15a section 8).
# (table, key column) pairs the catalogue handles by cascade from the photo or by explicit delete.
SUBJECT_KEY_COLUMNS = {
    'photo_key', 'item_key', 'box_key', 'face_key', 'entity_key', 'subject_key', 'image_id',
    'crop_id', 'face_id', 'member_id', 'subject_crop_id', 'subject_pair_id', 'subject_face_id',
    'subject_entity_key', 'person_item_key', 'person_entity_key', 'cover_photo_key',
}  # fmt: skip

CLAIM_KIND_COLUMNS = ('kind', 'subject_kind', 'layer', 'pins_geometry', 'asserts_object')

# Additive changes to tables that exist (15a section 2.1 and 2.2); the other changes to existing
# tables wait for the lane that rewrites their writers.
ADDED_COLUMNS = {'dataset_imports': 'dataset_name', 'label_operations': 'dataset_import_id'}
ADDED_FOREIGN_KEYS = {('label_operations', 'dataset_import_id', 'dataset_imports', 'n')}
ADDED_FOREIGN_KEY_INDEXES = {('label_operations', 'dataset_import_id', True)}

# What ``test_schema_matches_plan.py`` merges: every index the migration creates, by whether it is
# partial, and the foreign keys the catalog test reads (a composite key by its first column).
_ALL_FK_INDEXES = FOREIGN_KEY_INDEXES | ADDED_FOREIGN_KEY_INDEXES
ALL_PARTIAL_INDEXES = PARTIAL_INDEXES | {
    f'ix_{t}_{c}' for t, c, partial in _ALL_FK_INDEXES if partial
}
ALL_PLAIN_INDEXES = PLAIN_INDEXES | {
    f'ix_{t}_{c}' for t, c, partial in _ALL_FK_INDEXES if not partial
}
ALL_FOREIGN_KEYS = FOREIGN_KEYS | ADDED_FOREIGN_KEYS
