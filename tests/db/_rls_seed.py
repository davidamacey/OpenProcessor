"""One row in every row-level-secured table, for a project, written past row-level security.

The seed runs as the superuser. Ids are derived from the project id and a name so a test can
name them again. ``PROJECT_COLUMN`` says where each table keeps its project.
"""

from __future__ import annotations

import uuid
from typing import Any


# table -> column holding its project, or None for a child table that reaches it through a parent
PROJECT_COLUMN: dict[str, str | None] = {
    'project_indexes': 'project_id', 'class_registries': 'project_id',
    'project_classes': 'project_id', 'search_projection_tasks': 'project_id',
    'config_documents': 'project_id', 'config_activations': 'project_id',
    'config_activation_events': 'project_id', 'config_runtime_reports': 'project_id',
    'project_settings': 'project_id', 'project_keymaps': 'project_id',
    'project_state': 'project_id', 'projection_states': 'project_id', 'jobs': 'project_id',
    'events': 'project_id', 'stream_worker_claims': 'project_id',
    'dataset_uploads': 'project_id', 'dataset_imports': 'project_id',
    'ingest_sources': 'project_id', 'label_operations': 'project_id',
    'holdout_freezes': 'project_id', 'dataset_exports': 'project_id',
    'training_runs': 'project_id', 'models': 'project_id', 'promotions': 'project_id',
    'bakeoff_runs': 'project_id', 'combine_runs': 'target_project_id',
    'job_attempts': None, 'job_items': None, 'gpu_reservations': None,
    'dataset_import_images': None, 'combine_sources': None, 'ingest_source_files': None,
    'label_operation_members': None, 'config_revisions': None, 'model_engines': None,
}  # fmt: skip

_NS = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')

_JOB = (
    'INSERT INTO jobs (id, project_id, kind, queue, status, max_queue_wait, requested_by) '
    "VALUES (%({name})s, %(p)s, 'noop', 'jobs', 'queued', interval '1 hour', 'test')"
)


def ids(project: uuid.UUID, tag: str) -> dict[str, Any]:
    def make(name: str) -> uuid.UUID:
        return uuid.uuid5(_NS, f'{project}:{tag}:{name}')

    names = ('job', 'job2', 'job3', 'job4', 'doc', 'imp', 'src', 'op', 'model', 'run', 'exp', 'gpu')
    out: dict[str, Any] = {name: make(name) for name in names}
    out.update(p=project, k=f'{tag}-{str(project)[:8]}', gpu_uuid=f'GPU-{tag}-{str(project)[:8]}')
    return out


# (table, statement) in dependency order; %(p)s is the project id, %(k)s a unique tag.
def statements() -> list[tuple[str, str]]:
    def job(name: str) -> str:
        return _JOB.replace('{name}', name)

    return [
        ('project_indexes', "INSERT INTO project_indexes (project_id, role, index_name) VALUES (%(p)s, 'images', %(index_name)s)"),
        ('class_registries', 'INSERT INTO class_registries (project_id) VALUES (%(p)s)'),
        ('project_classes', "INSERT INTO project_classes (project_id, class_id, name, name_key) VALUES (%(p)s, 0, 'car', 'car')"),
        ('jobs', job('job')),
        ('search_projection_tasks', "INSERT INTO search_projection_tasks (project_id, kind, payload) VALUES (%(p)s, 'class_rename', '{}')"),
        ('config_documents', "INSERT INTO config_documents (id, project_id, kind, name, current_revision, updated_by) VALUES (%(doc)s, %(p)s, 'prompt_pack', 'pack', 1, 'test')"),
        ('config_revisions', "INSERT INTO config_revisions (config_id, revision, body, body_sha256, created_by) VALUES (%(doc)s, 1, '{}', 'x', 'test')"),
        ('config_activations', "INSERT INTO config_activations (project_id, axis, kind, config_id, revision, activated_by) VALUES (%(p)s, 'prompt_pack', 'prompt_pack', %(doc)s, 1, 'test')"),
        ('config_activation_events', "INSERT INTO config_activation_events (project_id, axis, to_config_id, to_revision, reason, by) VALUES (%(p)s, 'prompt_pack', %(doc)s, 1, 'activate', 'test')"),
        ('config_runtime_reports', "INSERT INTO config_runtime_reports (project_id, process, host, config_epoch) VALUES (%(p)s, 'api', 'h', 1)"),
        ('project_settings', "INSERT INTO project_settings (project_id, updated_by) VALUES (%(p)s, 'test')"),
        ('project_keymaps', "INSERT INTO project_keymaps (project_id, updated_by) VALUES (%(p)s, 'test')"),
        ('project_state', "INSERT INTO project_state (project_id, key, value) VALUES (%(p)s, 'cluster_refresh', '{}')"),
        ('projection_states', "INSERT INTO projection_states (project_id, projection_version, scope, n_points, n_components, metric, fitted_at) VALUES (%(p)s, 1, 'all', 1, 2, 'cosine', now())"),
        ('job_attempts', "INSERT INTO job_attempts (job_id, attempt, fencing_token, owner) VALUES (%(job)s, 1, 1, 'w')"),
        ('job_items', "INSERT INTO job_items (job_id, position, item_id) VALUES (%(job)s, 0, 'i')"),
        ('events', "INSERT INTO events (project_id, type, payload) VALUES (%(p)s, 'job.state', '{}')"),
        ('stream_worker_claims', "INSERT INTO stream_worker_claims (worker_kind, project_id, worker_id, lease_expires_at) VALUES ('detection', %(p)s, 'w', now())"),
        ('gpu_devices', "INSERT INTO gpu_devices (gpu_uuid, host, nvml_index, name, compute_capability, memory_total_mib, hw_jpeg_decoder, driver_version, cuda_version) VALUES (%(gpu_uuid)s, 'h', 0, 'g', '8.6', 1, false, 'd', 'c')"),
        ('gpu_reservations', 'INSERT INTO gpu_reservations (gpu_uuid, holder_job_id) VALUES (%(gpu_uuid)s, %(job)s)'),
        ('dataset_uploads', "INSERT INTO dataset_uploads (project_id, filename, size_bytes, storage_key, status, expires_at) VALUES (%(p)s, 'a.zip', 1, %(upload_key)s, 'received', now())"),
        ('dataset_imports', "INSERT INTO dataset_imports (id, project_id, format, mapping, options, source_key) VALUES (%(imp)s, %(p)s, 'yolo', '[]', '{}', 'k')"),
        ('dataset_import_images', "INSERT INTO dataset_import_images (import_id, image_key, status) VALUES (%(imp)s, 'a.jpg', 'pending')"),
        ('combine_runs', "INSERT INTO combine_runs (job_id, target_project_id, holdout_mode, class_mapping) VALUES (%(job)s, %(p)s, 'union', '{}')"),
        ('combine_sources', "INSERT INTO combine_sources (job_id, position, source_slug) VALUES (%(job)s, 0, 's')"),
        ('ingest_sources', "INSERT INTO ingest_sources (id, project_id, name, kind, config, schedule) VALUES (%(src)s, %(p)s, 'src', 'directory', '{}', '{}')"),
        ('ingest_source_files', "INSERT INTO ingest_source_files (source_id, file_key, status) VALUES (%(src)s, 'f', 'pending')"),
        ('label_operations', "INSERT INTO label_operations (id, project_id, kind, status, params, created_by) VALUES (%(op)s, %(p)s, 'k', 'written', '{}', 'test')"),
        ('label_operation_members', "INSERT INTO label_operation_members (op_id, item_id, prior) VALUES (%(op)s, 'i', '{}')"),
        ('holdout_freezes', "INSERT INTO holdout_freezes (project_id, is_current, source, item_count, frozen_by) VALUES (%(p)s, true, 'manual', 1, 'test')"),
        ('dataset_exports', "INSERT INTO dataset_exports (id, project_id, kind, class_names, manifest_sha256, artifact_prefix, split_seed) VALUES (%(exp)s, %(p)s, 'yolo', '{car}', 'x', 'a', 1)"),
        ('jobs', "INSERT INTO jobs (id, project_id, kind, queue, status, max_queue_wait, requested_by) VALUES (%(job2)s, %(p)s, 'train', 'gpu_train', 'queued', interval '1 hour', 'test')"),
        ('training_runs', "INSERT INTO training_runs (id, project_id, job_id, spec, class_names, family, base_model, artifact_prefix) VALUES (%(run)s, %(p)s, %(job2)s, '{}', '{car}', 'yolo', 'b', 'a')"),
        ('models', "INSERT INTO models (id, project_id, name, version, origin, family, task, labels, artifact_storage_key, created_by) VALUES (%(model)s, %(p)s, 'm', 1, 'trained', 'yolo', 'detect', '{car}', 'k', 'test')"),
        ('model_engines', "INSERT INTO model_engines (model_id, triton_model_name, precision, input_size, input_contract, output_contract, build_settings, build_stamp, max_batch, trt_version, gpu_arch, status) VALUES (%(model)s, %(triton_name)s, 'fp16', 640, '{}', '{}', '{}', 's', 8, '10', 'sm_86', 'ready')"),
        ('promotions', "INSERT INTO promotions (project_id, model_id, triton_model_name, triton_version, promoted_by) VALUES (%(p)s, %(model)s, %(triton_name)s, 1, 'test')"),
        ('jobs', "INSERT INTO jobs (id, project_id, kind, queue, status, max_queue_wait, requested_by) VALUES (%(job3)s, %(p)s, 'bakeoff', 'gpu_eval', 'queued', interval '1 hour', 'test')"),
        ('bakeoff_runs', "INSERT INTO bakeoff_runs (job_id, project_id, eval_dataset_key, models) VALUES (%(job3)s, %(p)s, 'k', '[]')"),
    ]  # fmt: skip


def seed_project(conn: Any, slug: str, tag: str = 'a') -> uuid.UUID:
    """Create a project and one row in every protected table, as the superuser."""
    project = uuid.uuid4()
    conn.execute('INSERT INTO project_slugs (slug) VALUES (%s)', (slug,))
    conn.execute(
        'INSERT INTO projects (id, slug, display_name, status, index_prefix, storage_prefix, '
        "model_prefix, mlflow_experiment, created_by) VALUES (%s, %s, %s, 'active', %s, %s, %s, %s, 'test')",
        (project, slug, slug, f'{slug}__', f'projects/{slug}', f'{slug}__', slug),
    )
    params = ids(project, tag)
    params.update(
        index_name=f'idx-{params["k"]}',
        upload_key=f'up/{params["k"]}',
        triton_name=f'm-{params["k"]}',
    )
    for _table, statement in statements():
        conn.execute(statement, params)
    return project
