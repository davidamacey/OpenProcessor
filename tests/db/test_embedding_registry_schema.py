"""The embedding registry: one primary space per target, immutable model versions."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
from _research_seed import make_model_version, make_project, make_space
from psycopg import errors


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db


@pytest.fixture
def conn(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        yield c


@pytest.fixture
def project(conn):
    return make_project(conn, f'p{uuid.uuid4().hex[:12]}')


@pytest.fixture
def version(conn):
    return make_model_version(conn, f'm{uuid.uuid4().hex[:10]}')


def test_a_second_primary_for_one_target_is_a_unique_violation(conn, project, version) -> None:
    make_space(conn, project, version, 'first_v1')
    with pytest.raises(errors.UniqueViolation, match='uq_embedding_spaces_one_primary'):
        make_space(conn, project, version, 'second_v1')
    # another target, or a non-primary role, is fine
    make_space(conn, project, version, 'second_v1', role='optional')
    make_space(conn, project, version, 'third_v1', target='item_text', vector_field='emb_text')


def test_a_primary_space_must_be_ready(conn, project, version) -> None:
    with pytest.raises(errors.CheckViolation, match='primary_is_ready'):
        make_space(conn, project, version, 'early_v1', status='building')


def test_a_box_space_needs_a_nested_path_and_a_frame_space_the_images_index(
    conn, project, version
) -> None:
    with pytest.raises(errors.CheckViolation, match='box_has_nested_path'):
        make_space(conn, project, version, 'box_v1', target='box', role='optional')
    with pytest.raises(errors.CheckViolation, match='frame_in_images'):
        conn.execute(
            'INSERT INTO embedding_spaces (project_id, space_key, model_version_id, target, dim, '
            "role, status, index_role, vector_field, knn_method, created_by) VALUES (%s, 'frame_v1', "
            "%s, 'frame', 512, 'optional', 'ready', 'items', 'f', '{}', 'test')",
            (project, version),
        )
    make_space(conn, project, version, 'frame_v2', target='frame', role='optional')


def test_a_model_version_refuses_an_input_spec_update(conn, version) -> None:
    with pytest.raises(errors.IntegrityConstraintViolation, match='only status and the parity'):
        conn.execute(
            """UPDATE embedding_model_versions SET input_spec = '{"x": 1}' WHERE id = %s""",
            (version,),
        )


def test_the_parity_is_recorded_once_and_the_status_moves(conn, version) -> None:
    conn.execute(
        'UPDATE embedding_model_versions SET engine_parity_min_cos = 0.998 WHERE id = %s',
        (version,),
    )
    with pytest.raises(errors.IntegrityConstraintViolation, match='recorded once'):
        conn.execute(
            'UPDATE embedding_model_versions SET engine_parity_min_cos = 0.9 WHERE id = %s',
            (version,),
        )
    conn.execute(
        "UPDATE embedding_model_versions SET status = 'deprecated' WHERE id = %s", (version,)
    )


def test_a_model_version_needs_its_triton_models_for_its_modalities(conn) -> None:
    with pytest.raises(errors.CheckViolation, match='text_model'):
        conn.execute(
            'INSERT INTO embedding_model_versions (model_key, version, modalities, native_dim, '
            'distance, input_spec, input_spec_sha256, weights_ref, weights_sha256, precision, '
            "triton_image_model, licence, status, created_by) VALUES ('textless', 1, "
            "ARRAY['image', 'text'], 512, 'cosine', '{}', 's', 'r', 'w', 'fp16', 'img', 'mit', "
            "'available', 'test')"
        )


def test_a_model_version_in_use_cannot_be_deleted(conn, project, version) -> None:
    make_space(conn, project, version, 'used_v1')
    with pytest.raises(errors.RestrictViolation):
        conn.execute('DELETE FROM embedding_model_versions WHERE id = %s', (version,))


def test_an_evaluation_of_a_space_against_itself_is_refused(conn, project, version) -> None:
    """The table has no 0.7.0 writer; this schema test stands in for its reader."""
    base = make_space(conn, project, version, 'base_v1')
    cand = make_space(conn, project, version, 'cand_v1', role='candidate', status='building')
    insert = (
        'INSERT INTO embedding_evaluation_runs (project_id, baseline_space_id, candidate_space_id, '
        'protocol, sample_spec, metrics, verdict, created_by) VALUES (%s, %s, %s, '
        "'label_consistency_v1', '{}', '{}', 'tie', 'test')"
    )
    conn.execute(insert, (project, base, cand))
    with pytest.raises(errors.CheckViolation, match='distinct_spaces'):
        conn.execute(insert, (project, base, base))
