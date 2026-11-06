"""Rows for the research-seam tests, written as the superuser (past row-level security).

Triggers still fire for the superuser, so the immutability and project-copy rules are exercised.
"""

from __future__ import annotations

import uuid
from typing import Any


def make_project(conn: Any, slug: str) -> uuid.UUID:
    project = uuid.uuid4()
    conn.execute('INSERT INTO project_slugs (slug) VALUES (%s)', (slug,))
    conn.execute(
        'INSERT INTO projects (id, slug, display_name, status, index_prefix, storage_prefix, '
        "model_prefix, mlflow_experiment, created_by) VALUES (%s, %s, %s, 'active', %s, %s, %s, %s, 'test')",
        (project, slug, slug, f'{slug}__', f'projects/{slug}', f'{slug}__', slug),
    )
    return project


def make_job(conn: Any, project: uuid.UUID | None, kind: str = 'noop') -> uuid.UUID:
    return conn.execute(
        'INSERT INTO jobs (project_id, kind, queue, status, max_queue_wait, requested_by) '
        "VALUES (%s, %s, 'jobs', 'queued', interval '1 hour', 'test') RETURNING id",
        (project, kind),
    ).fetchone()[0]


def make_profile(conn: Any, project: uuid.UUID, kind: str = 'prompt_pack') -> tuple[uuid.UUID, int]:
    """A config document with its first revision; returns ``(config_id, revision)``."""
    doc = conn.execute(
        'INSERT INTO config_documents (project_id, kind, name, current_revision, updated_by) '
        "VALUES (%s, %s, %s, 1, 'test') RETURNING id",
        (project, kind, f'doc-{uuid.uuid4().hex[:8]}'),
    ).fetchone()[0]
    conn.execute(
        'INSERT INTO config_revisions (config_id, revision, body, body_sha256, created_by) '
        "VALUES (%s, 1, '{}', 'x', 'test')",
        (doc,),
    )
    return doc, 1


def make_calibration(conn: Any, project: uuid.UUID | None, target: str = 'item') -> uuid.UUID:
    return conn.execute(
        'INSERT INTO cluster_calibrations (project_id, target, level, embedding_field, '
        'embedding_model, preprocess_id, kind, fitted_on, body, dev_metrics, created_by) '
        "VALUES (%s, %s, 'class', 'f', 'm', 'p', 'assign_bands', 'pin:x', '{}', '{}', 'test') "
        'RETURNING id',
        (project, target),
    ).fetchone()[0]


def make_partition(
    conn: Any,
    project: uuid.UUID | None,
    *,
    status: str = 'building',
    job: uuid.UUID | None = None,
    parent: uuid.UUID | None = None,
    calibration: uuid.UUID | None = None,
    profile: tuple[uuid.UUID, int] | None = None,
    target: str = 'item',
) -> uuid.UUID:
    config_id, revision = profile or (None, None)
    return conn.execute(
        'INSERT INTO cluster_partitions (project_id, target, level, embedding_field, '
        'embedding_model, preprocess_id, method, method_version, profile_config_id, '
        'profile_revision, params, calibration_id, pool_filter, input_digest, seed, backend, '
        'backend_detail, parent_partition_id, job_id, status, created_by) '
        "VALUES (%s, %s, 'class', 'f', 'm', 'p', 'ivf', '1', %s, %s, '{}', %s, '{}', 'd', 1, "
        "'cpu', '{}', %s, %s, %s, 'test') RETURNING id",
        (project, target, config_id, revision, calibration, parent, job, status),
    ).fetchone()[0]


def make_identity(conn: Any, project: uuid.UUID | None, target: str = 'item') -> uuid.UUID:
    return conn.execute(
        "INSERT INTO cluster_identities (project_id, target, level) VALUES (%s, %s, 'class') "
        'RETURNING cluster_uid',
        (project, target),
    ).fetchone()[0]


def add_cluster(conn: Any, partition: uuid.UUID, uid: uuid.UUID, display_no: int = 1) -> None:
    conn.execute(
        'INSERT INTO partition_clusters (partition_id, cluster_uid, display_no, transition, size) '
        "VALUES (%s, %s, %s, 'new', 3)",
        (partition, uid, display_no),
    )


def add_membership(conn: Any, partition: uuid.UUID, key: str, uid: uuid.UUID | None = None) -> None:
    state = 'assigned' if uid else 'unassigned'
    conn.execute(
        'INSERT INTO cluster_memberships (partition_id, item_key, cluster_uid, state) '
        'VALUES (%s, %s, %s, %s)',
        (partition, key, uid, state),
    )


def ready_partition(conn: Any, project: uuid.UUID | None, **kwargs: Any) -> uuid.UUID:
    """A partition with one cluster and two memberships, then marked ready."""
    partition = make_partition(conn, project, **kwargs)
    uid = make_identity(conn, project)
    add_cluster(conn, partition, uid)
    add_membership(conn, partition, 'a', uid)
    add_membership(conn, partition, 'b')
    conn.execute("UPDATE cluster_partitions SET status = 'ready' WHERE id = %s", (partition,))
    return partition


def bind_scope(conn: Any, kind: str, project: uuid.UUID | None = None) -> None:
    """Open the row-level-security scope of the current transaction, as ``project_tx`` does."""
    conn.execute("SELECT set_config('op.scope', %s, true)", (kind,))
    conn.execute("SELECT set_config('op.project_id', %s, true)", (str(project or ''),))


def make_model_version(conn: Any, key: str = 'pe_core', version: int = 1) -> uuid.UUID:
    return conn.execute(
        'INSERT INTO embedding_model_versions (model_key, version, modalities, native_dim, distance, '
        'input_spec, input_spec_sha256, weights_ref, weights_sha256, precision, triton_image_model, '
        "triton_text_model, licence, status, created_by) VALUES (%s, %s, ARRAY['image', 'text'], "
        "1024, 'cosine', '{}', 'sha', 'hf:x@y', 'w', 'fp16', 'img', 'txt', 'apache-2.0', "
        "'available', 'test') RETURNING id",
        (key, version),
    ).fetchone()[0]


def make_space(
    conn: Any,
    project: uuid.UUID,
    model_version: uuid.UUID,
    space_key: str = 'pe_core_v1',
    *,
    target: str = 'item',
    role: str = 'primary',
    status: str = 'ready',
    vector_field: str | None = None,
) -> uuid.UUID:
    field = vector_field or f'emb_{space_key}'
    return conn.execute(
        'INSERT INTO embedding_spaces (project_id, space_key, model_version_id, target, dim, role, '
        'status, index_role, vector_field, knn_method, created_by) VALUES (%s, %s, %s, %s, 1024, '
        "%s, %s, %s, %s, '{}', 'test') RETURNING id",
        (
            project,
            space_key,
            model_version,
            target,
            role,
            status,
            'images' if target == 'frame' else 'items',
            field,
        ),
    ).fetchone()[0]


def make_loop_run(conn: Any, project: uuid.UUID, job: uuid.UUID | None = None) -> uuid.UUID:
    return conn.execute(
        'INSERT INTO assist_loop_runs (project_id, job_id, request, strategy, seed, pinned, '
        "created_by) VALUES (%s, %s, '{}', 'cluster_sampled', 7, '{}', 'test') RETURNING id",
        (project, job or make_job(conn, project, 'assist_loop')),
    ).fetchone()[0]
