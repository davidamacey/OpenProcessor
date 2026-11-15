"""Machine results of revision 0010 (15a section 2.3): capture, detections, opinions, boxes, scores."""

from __future__ import annotations

from src.db.migrations.schema_0010.ddl import IDENTITY, PROJECT, Table, ck, closed_set, pk, uq


LEDGER_STATES = ('pending', 'done', 'failed', 'skipped')
OPINION_SOURCES = (
    'classifier', 'vlm', 'assist_propagated', 'import_suggestion', 'adopted_detector',
    'open_vocab_target',
)  # fmt: skip
OPINION_OUTCOMES = (
    'matched', 'unmatched', 'new_class_pending', 'reclassified', 'empty', 'low_conf', 'model',
)  # fmt: skip
BOX_STATES = ('proposed', 'accepted', 'rejected', 'false_positive')
NOTE_KINDS = ('proposal', 'class_mismatch', 'missed_label', 'unmatched_detection')
_ANALYSER = 'analyser_id integer NOT NULL -> analysers(analyser_id)'
_ITEM = 'item_key bigint NOT NULL -> items(item_key) CASCADE'
_PHOTO = 'photo_key bigint NOT NULL -> photos(photo_key) CASCADE'

ANALYSIS_LEDGER = Table(
    'analysis_ledger',
    (
        _PHOTO,
        PROJECT,
        'kind text NOT NULL',
        _ANALYSER,
        'state text NOT NULL',
        'reason text NULL',
        'updated_at timestamptz NOT NULL DEFAULT now()',
    ),
    (pk('photo_key', 'kind'), ck('state', closed_set('state', LEDGER_STATES))),
)

PHOTO_CAPTURE = Table(
    'photo_capture',
    (
        _PHOTO,
        PROJECT,
        _ANALYSER,
        'taken_local timestamp NULL',
        'taken_utc timestamp NULL',
        'offset_minutes smallint NULL',
        'time_source text NULL',
        'offset_source text NULL',
        'time_precision text NULL',
        'camera_make text NULL',
        'camera_model text NULL',
        'lens_model text NULL',
        'focal_length_mm real NULL',
        'focal_length_35mm real NULL',
        'orientation smallint NULL',
        'exif_status text NOT NULL',
        'has_location boolean NOT NULL DEFAULT false',
        'lat double precision NULL',
        'lon double precision NULL',
        'altitude_m real NULL',
        'location_accuracy_m real NULL',
        'heading_deg real NULL',
        'location_precision text NULL',
        'burst_id text NULL',
    ),
    (pk('photo_key'), ck('position_pair', '(lat IS NULL) = (lon IS NULL)')),
)

DETECTIONS = Table(
    'detections',
    (
        f'detection_key bigint {IDENTITY}',
        PROJECT,
        _PHOTO,
        _ITEM,
        _ANALYSER,
        'bbox real[] NOT NULL',
        'class_name text NULL',
        'class_id integer NULL',
        'confidence real NULL',
        'proposal_name text NULL',
        'blur_lap_var real NULL',
        'blur_lap_ratio real NULL',
        'blur_full_var real NULL',
        'prompt text NULL',
        'mask_polygon real[] NULL',
        'import_id uuid NULL -> dataset_imports(id) CASCADE',
        'created_at timestamptz NOT NULL DEFAULT now()',
    ),
    (
        pk('detection_key'),
        uq('uq_detections_item_analyser', 'item_key', 'analyser_id'),
        ck('bbox', 'cardinality(bbox) = 4'),
    ),
)

CLASS_OPINIONS = Table(
    'class_opinions',
    (
        _ITEM,
        PROJECT,
        'source text NOT NULL',
        _ANALYSER,
        'class_name text NULL',
        'class_id_hint integer NULL',
        'outcome text NOT NULL',
        'confidence real NULL',
        'confidence_cat text NULL',
        'raw_label text NULL',
        'raw_label_conf real NULL',
        'raw_class text NULL',
        'proposed_class text NULL',
        'empty_reason text NULL',
        'via text NULL',
        'op_id uuid NULL',
        'import_id uuid NULL -> dataset_imports(id) CASCADE',
        'created_at timestamptz NOT NULL DEFAULT now()',
    ),
    (
        pk('item_key', 'source'),
        ck('source', closed_set('source', OPINION_SOURCES)),
        ck('outcome', closed_set('outcome', OPINION_OUTCOMES)),
        ck('import_source', "(source = 'import_suggestion') = (import_id IS NOT NULL)"),
        ck(
            'operation_source',
            "source NOT IN ('assist_propagated', 'adopted_detector') OR op_id IS NOT NULL",
        ),
    ),
)

RAW_LABEL_CLUSTERS = Table(
    'raw_label_clusters',
    (
        PROJECT,
        'raw_label text NOT NULL',
        _ANALYSER,
        'cluster_id integer NOT NULL',
        'cluster_name text NOT NULL',
        'distance real NULL',
    ),
    (pk('project_id', 'raw_label'),),
)

ITEM_REGION_RESULTS = Table(
    'item_region_results',
    (
        _ITEM,
        PROJECT,
        _ANALYSER,
        'status text NOT NULL',
        'reason text NULL',
        'verified boolean NULL',
        'verified_at timestamptz NULL',
        'verifier text NULL',
        'verifier_version text NULL',
        'auto_confirmed boolean NULL',
        'visible boolean NULL',
        'skip_verify boolean NULL',
        'gate_skip text NULL',
        "detector_chain text[] NOT NULL DEFAULT '{}'",
        'set_complete boolean NULL',
        'detected_at timestamptz NOT NULL DEFAULT now()',
    ),
    (pk('item_key'),),
)

REGION_BOXES = Table(
    'region_boxes',
    (
        f'box_key bigint {IDENTITY}',
        PROJECT,
        _ITEM,
        'box_id text NOT NULL',
        _ANALYSER,
        'region_seq integer NOT NULL',
        'bbox real[] NOT NULL',
        'state text NOT NULL',
        'score real NULL',
        'detector text NULL',
        'detector_version text NULL',
        'source text NULL',
        'confidence text NULL',
        'rejection_reason text NULL',
        'detected_at timestamptz NOT NULL DEFAULT now()',
        'import_id uuid NULL -> dataset_imports(id) CASCADE',
    ),
    (
        pk('box_key'),
        uq('uq_region_boxes_item_box', 'item_key', 'box_id'),
        ck('bbox', 'cardinality(bbox) = 4'),
        ck('state', closed_set('state', BOX_STATES)),
    ),
)

BOX_TEXTS = Table(
    'box_texts',
    (
        'box_key bigint NOT NULL -> region_boxes(box_key) CASCADE',
        PROJECT,
        'reader text NOT NULL',
        _ANALYSER,
        'text text NULL',
        'text_raw text NULL',
        'text_frame text NULL',
        'invalid text NULL',
        'confidence real NULL',
        'conf_char_min real NULL',
        'jurisdiction text NULL',
        'jurisdiction_conf real NULL',
    ),
    (pk('box_key', 'reader'), ck('reader', "reader IN ('ocr', 'vlm')")),
)

TEXT_LINES = Table(
    'text_lines',
    (
        f'line_key bigint {IDENTITY}',
        PROJECT,
        _PHOTO,
        'item_key bigint NULL -> items(item_key) CASCADE',
        _ANALYSER,
        'seq smallint NOT NULL',
        'text text NOT NULL',
        'text_raw text NULL',
        'box_norm real[] NOT NULL',
        'polygon real[] NULL',
        'det_score real NULL',
        'rec_score real NULL',
        'rel_height real NULL',
        'language text NULL',
    ),
    (pk('line_key'),),
)

PROBE_PREDICTIONS = Table(
    'probe_predictions',
    (
        _ITEM,
        PROJECT,
        _ANALYSER,
        'class_name text NULL',
        'class_id_hint integer NULL',
        'entropy real NULL',
        'confidence real NULL',
        'margin real NULL',
        'scored_at timestamptz NOT NULL DEFAULT now()',
    ),
    (pk('item_key'),),
)

ITEM_SCORES = Table(
    'item_scores',
    (
        _ITEM,
        PROJECT,
        'kind text NOT NULL',
        _ANALYSER,
        'score real NOT NULL',
        'scored_at timestamptz NOT NULL DEFAULT now()',
    ),
    (pk('item_key', 'kind'), ck('kind', "kind IN ('uniqueness', 'mistakenness')")),
)

ITEM_PROJECTIONS = Table(
    'item_projections',
    (_ITEM, PROJECT, _ANALYSER, 'x real NOT NULL', 'y real NOT NULL'),
    (pk('item_key'),),
)

ASSIST_ANSWERS = Table(
    'assist_answers',
    (
        _ITEM,
        PROJECT,
        _ANALYSER,
        'run_id uuid NOT NULL',
        'run_created_at timestamptz NOT NULL',
        'class_name text NULL',
        'class_confidence text NULL',
        "attributes jsonb NOT NULL DEFAULT '[]'",
        "box_attributes jsonb NOT NULL DEFAULT '[]'",
    ),
    (pk('item_key'),),
)

ITEM_NOTES = Table(
    'item_notes',
    (
        f'note_key bigint {IDENTITY}',
        PROJECT,
        _ITEM,
        'kind text NOT NULL',
        _ANALYSER,
        'import_id uuid NULL -> dataset_imports(id) CASCADE',
        'body jsonb NOT NULL',
        'created_at timestamptz NOT NULL DEFAULT now()',
    ),
    (pk('note_key'), ck('kind', closed_set('kind', NOTE_KINDS))),
)

FACES = Table(
    'faces',
    (
        f'face_key bigint {IDENTITY}',
        PROJECT,
        _PHOTO,
        'face_id uuid NOT NULL',
        _ANALYSER,
        'person_item_key bigint NULL -> items(item_key) SET NULL',
        'bbox real[] NOT NULL',
        'landmarks real[] NOT NULL',
        'det_score real NOT NULL',
        'quality real NULL',
        'yaw_deg real NULL',
        'eyes_open boolean NULL',
        'side_px integer NULL',
    ),
    (
        pk('face_key'),
        uq('uq_faces_face_id', 'project_id', 'face_id'),
        ck('landmarks', 'cardinality(landmarks) = 10'),
    ),
)

TABLES = (
    ANALYSIS_LEDGER, PHOTO_CAPTURE, DETECTIONS, CLASS_OPINIONS, RAW_LABEL_CLUSTERS,
    ITEM_REGION_RESULTS, REGION_BOXES, BOX_TEXTS, TEXT_LINES, PROBE_PREDICTIONS, ITEM_SCORES,
    ITEM_PROJECTIONS, ASSIST_ANSWERS, ITEM_NOTES, FACES,
)  # fmt: skip
