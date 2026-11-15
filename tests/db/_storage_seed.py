"""One row in every table of the storage record for a project, written as the superuser.

Rows are valid (every key, check and foreign key holds), so a test can read, update or delete them as
the application role and a policy, not a constraint, is what answers. ``seed_world`` returns the ids a
test names again.
"""

from __future__ import annotations

import uuid
from typing import Any

import _storage_plan as plan
from _research_seed import make_model_version, make_project, make_space


NS = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
BOX = '{0.1,0.1,0.5,0.5}'
ANALYSER_DIGEST = 'c' * 64


def scalar(conn: Any, sql: str, params: tuple[Any, ...] = ()) -> Any:
    """The first column of the one row ``sql`` returns."""
    row = conn.execute(sql, params).fetchone()
    assert row is not None
    return row[0]


def wire(project: uuid.UUID, name: str) -> uuid.UUID:
    return uuid.uuid5(NS, f'{project}:{name}')


def analyser(conn: Any, kind: str = 'detector', name: str = 'primary', version: str = '1') -> int:
    conn.execute(
        'INSERT INTO analysers (kind, name, version, config_sha256) VALUES (%s, %s, %s, %s) '
        'ON CONFLICT DO NOTHING',
        (kind, name, version, ANALYSER_DIGEST),
    )
    return conn.execute(
        'SELECT analyser_id FROM analysers WHERE kind = %s AND name = %s AND version = %s '
        'AND config_sha256 = %s',
        (kind, name, version, ANALYSER_DIGEST),
    ).fetchone()[0]


def photo(conn: Any, project: uuid.UUID, name: str, *, frame: int = 1) -> int:
    return conn.execute(
        'INSERT INTO photos (project_id, image_id, imohash, storage_key, size_bytes, width, height, '
        "frame, decode_route, source) VALUES (%s, %s, %s, %s, 10, 640, 480, %s, 'gpu', 'test') "
        'RETURNING photo_key',
        (project, wire(project, name), bytes([len(name)]) * 16, f'root/{name}.jpg', frame),
    ).fetchone()[0]


def item(conn: Any, project: uuid.UUID, photo_key: int, name: str) -> int:
    return conn.execute(
        'INSERT INTO items (project_id, photo_key, crop_id) VALUES (%s, %s, %s) RETURNING item_key',
        (project, photo_key, wire(project, name)),
    ).fetchone()[0]


def detection(
    conn: Any,
    project: uuid.UUID,
    photo_key: int,
    item_key: int,
    analyser_id: int,
    *,
    box: str = BOX,
    class_name: str = 'car',
) -> None:
    conn.execute(
        'INSERT INTO detections (project_id, photo_key, item_key, analyser_id, bbox, class_name, '
        'confidence) VALUES (%s, %s, %s, %s, %s::real[], %s, 0.9)',
        (project, photo_key, item_key, analyser_id, box, class_name),
    )


def seed_world(conn: Any, slug: str) -> dict[str, Any]:
    """A project with one row in each of the 35 tables of the plan (the two catalogs are shared)."""
    project = make_project(conn, slug)
    det = analyser(conn)
    ph = photo(conn, project, 'p1')
    it = item(conn, project, ph, 'i1')
    ent = conn.execute(
        "INSERT INTO entities (project_id, kind) VALUES (%s, 'person') RETURNING entity_key",
        (project,),
    ).fetchone()[0]
    conn.execute(
        'INSERT INTO photo_provenance (photo_key, project_id, kind) VALUES (%s, %s, %s)',
        (ph, project, 'combine'),
    )
    claim = conn.execute(
        'INSERT INTO claims (project_id, actor_kind, actor, op_id, kind, subject_kind, photo_key, '
        'subject_crop_id, subject_bbox, subject_frame, payload) VALUES (%s, %s, %s, %s, %s, %s, %s, '
        '%s, %s::real[], 1, %s) RETURNING claim_id',
        (project, 'human', 'api', wire(project, 'op'), 'class_set', 'item', ph, wire(project, 'i1'), BOX,
         '{"class_name": "car", "label_source": "human", "via": "single", "labeler": "human"}'),
    ).fetchone()[0]  # fmt: skip
    conn.execute(
        'INSERT INTO claims (project_id, actor_kind, actor, kind, subject_kind, withdraws_claim_id) '
        "VALUES (%s, 'human', 'api', 'withdraw', 'claim', %s)",
        (project, claim),
    )
    conn.execute(
        "INSERT INTO analysis_ledger (photo_key, project_id, kind, analyser_id, state) VALUES (%s, %s, 'detector', %s, 'done')",
        (ph, project, det),
    )
    conn.execute(
        "INSERT INTO photo_capture (photo_key, project_id, analyser_id, exif_status) VALUES (%s, %s, %s, 'ok')",
        (ph, project, det),
    )
    detection(conn, project, ph, it, det)
    conn.execute(
        "INSERT INTO class_opinions (item_key, project_id, source, analyser_id, class_name, outcome) VALUES (%s, %s, 'vlm', %s, 'car', 'matched')",
        (it, project, det),
    )
    conn.execute(
        "INSERT INTO raw_label_clusters (project_id, raw_label, analyser_id, cluster_id, cluster_name) VALUES (%s, 'sedan', %s, 1, 'car')",
        (project, det),
    )
    conn.execute(
        "INSERT INTO item_region_results (item_key, project_id, analyser_id, status) VALUES (%s, %s, %s, 'detected')",
        (it, project, det),
    )
    box = conn.execute(
        'INSERT INTO region_boxes (project_id, item_key, box_id, analyser_id, region_seq, bbox, state) '
        "VALUES (%s, %s, 'b1', %s, 1, %s::real[], 'proposed') RETURNING box_key",
        (project, it, det, BOX),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO box_texts (box_key, project_id, reader, analyser_id, text) VALUES (%s, %s, 'ocr', %s, 'abc')",
        (box, project, det),
    )
    conn.execute(
        "INSERT INTO text_lines (project_id, photo_key, analyser_id, seq, text, box_norm) VALUES (%s, %s, %s, 0, 'abc', %s::real[])",
        (project, ph, det, BOX),
    )
    conn.execute(
        'INSERT INTO probe_predictions (item_key, project_id, analyser_id) VALUES (%s, %s, %s)',
        (it, project, det),
    )
    conn.execute(
        "INSERT INTO item_scores (item_key, project_id, kind, analyser_id, score) VALUES (%s, %s, 'uniqueness', %s, 0.5)",
        (it, project, det),
    )
    conn.execute(
        'INSERT INTO item_projections (item_key, project_id, analyser_id, x, y) VALUES (%s, %s, %s, 0.1, 0.2)',
        (it, project, det),
    )
    conn.execute(
        'INSERT INTO assist_answers (item_key, project_id, analyser_id, run_id, run_created_at) VALUES (%s, %s, %s, %s, now())',
        (it, project, det, uuid.uuid4()),
    )
    conn.execute(
        "INSERT INTO item_notes (project_id, item_key, kind, analyser_id, body) VALUES (%s, %s, 'proposal', %s, '{}')",
        (project, it, det),
    )
    face = conn.execute(
        'INSERT INTO faces (project_id, photo_key, face_id, analyser_id, bbox, landmarks, det_score) '
        'VALUES (%s, %s, %s, %s, %s::real[], %s::real[], 0.9) RETURNING face_key',
        (project, ph, wire(project, 'f1'), det, BOX, '{' + ','.join(['0.5'] * 10) + '}'),
    ).fetchone()[0]
    space = make_space(
        conn, project, make_model_version(conn, f'm{slug[:8]}'.replace('-', '_')), 'pe_core_v1'
    )
    conn.execute(
        "INSERT INTO vector_gaps (space_id, project_id, subject_key, reason) VALUES (%s, %s, %s, 'failed')",
        (space, project, it),
    )
    conn.execute(
        'INSERT INTO item_state (item_key, project_id, photo_key, crop_id, bbox, area, rank, '
        "geometry_source, geometry_revision, region_revision, created_at, updated_at) VALUES (%s, %s, %s, %s, %s::real[], 0.16, 1, 'detector', 0, 0, now(), now())",
        (it, project, ph, wire(project, 'i1'), BOX),
    )
    conn.execute(
        "INSERT INTO box_state (project_id, item_key, box_id, box_key, bbox, state, ordinal) VALUES (%s, %s, 'b1', %s, %s::real[], 'proposed', 0)",
        (project, it, box, BOX),
    )
    conn.execute(
        "INSERT INTO entity_edges (entity_key, project_id, subject_kind, subject_key, analyser_id) VALUES (%s, %s, 'photo', %s, %s)",
        (ent, project, ph, det),
    )
    conn.execute(
        "INSERT INTO entity_state (entity_key, project_id, kind, state, revision, updated_at) VALUES (%s, %s, 'person', 'active', 1, now())",
        (ent, project),
    )
    conn.execute(
        "INSERT INTO membership_state (entity_key, project_id, subject_kind, subject_key, source) VALUES (%s, %s, 'photo', %s, 'human')",
        (ent, project, ph),
    )
    conn.execute(
        "INSERT INTO face_state (face_key, project_id, person_state) VALUES (%s, %s, 'unassigned')",
        (face, project),
    )
    conn.execute(
        'INSERT INTO photo_state (photo_key, project_id, revision, updated_at) VALUES (%s, %s, 1, now())',
        (ph, project),
    )
    conn.execute(
        "INSERT INTO photo_files (photo_key, project_id, role, storage_key, size_bytes) VALUES (%s, %s, 'sidecar', 'root/p1.xmp', 3)",
        (ph, project),
    )
    conn.execute(
        'INSERT INTO photo_derivatives (photo_key, project_id, analyser_id, aspect) VALUES (%s, %s, %s, 1.3)',
        (ph, project, det),
    )
    conn.execute(
        'INSERT INTO photo_signals (photo_key, project_id, analyser_id) VALUES (%s, %s, %s)',
        (ph, project, det),
    )
    conn.execute(
        'INSERT INTO photo_places (photo_key, project_id, analyser_id) VALUES (%s, %s, %s)',
        (ph, project, det),
    )
    erasure = conn.execute(
        "INSERT INTO erasure_jobs (project_id, subject_kind, photo_keys, state) VALUES (%s, 'photo', %s, 'running') RETURNING id",
        (project, [ph]),
    ).fetchone()[0]
    return {
        'project': project, 'photo': ph, 'item': it, 'entity': ent, 'claim': claim, 'box': box,
        'face': face, 'space': space, 'analyser': det, 'erasure': erasure,
    }  # fmt: skip


def assert_every_table_seeded(conn: Any) -> None:
    empty = [
        t for t in sorted(plan.TABLES)
        if conn.execute(f'SELECT count(*) FROM {t}').fetchone()[0] == 0  # nosec B608  plan names
    ]  # fmt: skip
    assert not empty, f'the seed left these tables empty: {empty}'
