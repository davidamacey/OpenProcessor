"""The jobs, runs, sources, imports and files each project owns, so the read routes of those
families are reached with the bound project's own ids (and a foreign id can be refused).

Rows are written as the superuser past row-level security, with the shapes the services read;
the files sit where the routes read them. Every row and file is named with the project's slug.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any

from psycopg.types.json import Jsonb

from src.domain.storage_keys import make_key


if TYPE_CHECKING:
    from pathlib import Path

BAKEOFF_JOB = '{slug}-job-bakeoff'
BAKEOFF_DATASET = 'export:sweep'
LOOP_NODE = 'r10000'


def finished_job(
    conn: Any,
    project: uuid.UUID,
    kind: str,
    *,
    queue: str = 'jobs',
    status: str = 'completed',
    params: dict[str, Any] | None = None,
    error: str | None = None,
    result: dict[str, Any] | None = None,
) -> uuid.UUID:
    """A finished job (finished, so it never counts as the project's live one)."""
    row = conn.execute(
        'INSERT INTO jobs (project_id, kind, queue, status, params, max_queue_wait, requested_by, '
        'started_at, finished_at, error_message, result) '
        "VALUES (%s, %s, %s, %s, %s, interval '1 hour', 'test', now(), now(), %s, %s) RETURNING id",
        (
            project,
            kind,
            queue,
            status,
            Jsonb(params or {}),
            error,
            None if result is None else Jsonb(result),
        ),
    ).fetchone()
    assert row is not None
    return row[0]


def _partition(conn: Any, project: uuid.UUID) -> uuid.UUID:
    """A ready partition with one cluster and two memberships."""
    partition = conn.execute(
        'INSERT INTO cluster_partitions (project_id, target, level, embedding_field, '
        'embedding_model, preprocess_id, method, method_version, params, pool_filter, '
        'input_digest, seed, backend, backend_detail, status, created_by) '
        "VALUES (%s, 'item', 'class', 'f', 'm', 'p', 'ivf', '1', '{}', '{}', 'd', 1, 'cpu', "
        "'{}', 'building', 'test') RETURNING id",
        (project,),
    ).fetchone()[0]
    uid = conn.execute(
        "INSERT INTO cluster_identities (project_id, target, level) VALUES (%s, 'item', 'class') "
        'RETURNING cluster_uid',
        (project,),
    ).fetchone()[0]
    conn.execute(
        'INSERT INTO partition_clusters (partition_id, cluster_uid, display_no, transition, size) '
        "VALUES (%s, %s, 1, 'new', 3)",
        (partition, uid),
    )
    for key, owner in (('a', uid), ('b', None)):
        conn.execute(
            'INSERT INTO cluster_memberships (partition_id, item_key, cluster_uid, state) '
            'VALUES (%s, %s, %s, %s)',
            (partition, key, owner, 'assigned' if owner else 'unassigned'),
        )
    conn.execute("UPDATE cluster_partitions SET status = 'ready' WHERE id = %s", (partition,))
    return partition


def _import(conn: Any, project: uuid.UUID, slug: str) -> uuid.UUID:
    job = finished_job(conn, project, 'dataset_import')
    row = conn.execute(
        'INSERT INTO dataset_imports (project_id, job_id, format, mapping, options, source_key, '
        'scan, report) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id',
        (
            project,
            job,
            'yolo',
            Jsonb({}),
            Jsonb({}),
            f'{slug}-label-import',
            Jsonb({'import_key': f'{slug}-key', 'name': f'{slug}-label-import', 'images_total': 1}),
            Jsonb({}),
        ),
    ).fetchone()
    assert row is not None
    conn.execute(
        'INSERT INTO dataset_import_images (import_id, image_key, status, detail) '
        "VALUES (%s, %s, 'written', %s)",
        (row[0], f'{slug}-img-0001.jpg', Jsonb({'source_stem': f'{slug}-img-0001'})),
    )
    return row[0]


def _source(conn: Any, project: uuid.UUID, slug: str, path: str) -> uuid.UUID:
    row = conn.execute(
        'INSERT INTO ingest_sources (project_id, name, kind, config, schedule) '
        "VALUES (%s, %s, 'folder', %s, '{}') RETURNING id",
        (project, f'{slug}-label-source', Jsonb({'path': path})),
    ).fetchone()
    assert row is not None
    return row[0]


def _train(conn: Any, project: uuid.UUID, slug: str) -> tuple[uuid.UUID, uuid.UUID]:
    """A finished training run (with a manifest) and the promote job that failed on it."""
    job = finished_job(conn, project, 'train', queue='gpu_train')
    conn.execute(
        'INSERT INTO training_runs (project_id, job_id, spec, class_names, family, base_model, '
        'artifact_prefix, manifest) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)',
        (
            project,
            job,
            Jsonb({}),
            [f'{slug}_class'],
            'yolo',
            'b',
            f'projects/{slug}/training/{job}',
            Jsonb({'results': {}}),
        ),
    )
    promote = finished_job(
        conn,
        project,
        'promote',
        queue='gpu_train',
        status='failed',
        params={'run_job_id': str(job), 'triton_name': f'{slug}__model'},
        error=f'{slug}-promote-failed',
    )
    return job, promote


def _loop(conn: Any, project: uuid.UUID, slug: str) -> uuid.UUID:
    """A finished assist loop (no levels run), as the loop service reads it."""
    from src.schemas.curation.assist_loop import (
        AssistLoopRequest,
        LoopBudget,
        LoopPackPin,
        LoopPinned,
        LoopPlan,
        LoopVlmPin,
    )

    request = AssistLoopRequest(dry_run=False, budget=LoopBudget(max_vlm_crops=100))
    pinned = LoopPinned(
        vlm=LoopVlmPin(name=f'{slug}-model', revision=1, model='m'),
        prompt_pack=LoopPackPin(name=f'{slug}-model', revision=1),
    ).model_dump(mode='json')
    pinned['plan'] = LoopPlan(
        roots=0,
        eligible_items=0,
        required_sample_table=[],
        estimated_vlm_crops_root_level=0,
        estimated_vlm_requests_root_level=0,
    ).model_dump(mode='json')
    job = finished_job(conn, project, 'assist_loop')
    row = conn.execute(
        'INSERT INTO assist_loop_runs (project_id, job_id, request, strategy, seed, pinned, '
        "created_by) VALUES (%s, %s, %s, 'cluster_sampled', 7, %s, 'test') RETURNING id",
        (project, job, Jsonb(request.model_dump(mode='json')), Jsonb(pinned)),
    ).fetchone()
    assert row is not None
    _loop_node(conn, row[0], slug)
    return row[0]


def loop_node_ids(slug: str) -> list[str]:
    return [f'{slug}-item-0001', f'{slug}-item-0002']


def _loop_node(conn: Any, run: uuid.UUID, slug: str) -> None:
    """A level-1 node of the loop that ran: the row names the blob that
    ``seed_world_files`` writes under the same key (``SnapshotStore.key``)."""
    conn.execute(
        'INSERT INTO assist_loop_iterations (run_id, ordinal, status) VALUES (%s, 1, %s)',
        (run, 'open'),
    )
    key = make_key('projects', slug, 'assist_loop', str(run), 'nodes', f'{LOOP_NODE}.json')
    ids = loop_node_ids(slug)
    conn.execute(
        'INSERT INTO assist_loop_nodes (run_id, path, iteration, parent_path, level, member_count, '
        'eligible_count, membership_digest, snapshot_storage_key, step) '
        "VALUES (%s, %s, 1, 'r1', 1, %s, %s, 'sweep', %s, 'open')",
        (run, LOOP_NODE, len(ids), len(ids), key),
    )


def _assist_run(conn: Any, project: uuid.UUID, slug: str) -> uuid.UUID:
    from src.services.curation.assist.models import AssistRunRequest, CropsTarget, ResolvedScope

    request = AssistRunRequest(
        target=CropsTarget(kind='crops', crop_ids=[f'{slug}-item-0001']), actions=['classify']
    )
    resolved = ResolvedScope(scope_count=1, scope_digest='x', sampled=1, sample_seed=0)
    return finished_job(
        conn,
        project,
        'assist_run',
        params={
            'request': request.model_dump(mode='json'),
            'resolved': resolved.model_dump(mode='json'),
        },
    )


def seed_world(conn: Any, slug: str, project: uuid.UUID, source_dir: str) -> dict[str, str]:
    """Ids of the rows one project owns, keyed by what ``PG_ROW_IDS`` names them."""
    train_job, promote = _train(conn, project, slug)
    source = _source(conn, project, slug, source_dir)
    conn.execute(
        'INSERT INTO projection_states (project_id, projection_version, scope, n_points, '
        "n_components, metric, fitted_at) VALUES (%s, 1, 'all', 1, 2, 'cosine', now())",
        (project,),
    )
    return {
        'reprocess_job_id': str(finished_job(conn, project, 'reprocess')),
        'autolabel_job_id': str(finished_job(conn, project, 'autolabel')),
        'selection_job_id': str(
            finished_job(
                conn,
                project,
                'selection',
                result={
                    'selection': {
                        'crop_ids': [f'{slug}-item-0004'],
                        'method': 'kcenter_greedy',
                        'version': 'v1',
                        'n_pool': 1,
                    }
                },
            )
        ),
        'bulk_run_id': str(
            finished_job(
                conn,
                project,
                'bulk_ingest_run',
                params={
                    'source': {'kind': 'folder', 'path': source_dir},
                    'options': {},
                    'source_id': str(source),
                },
            )
        ),
        'loop_id': str(_loop(conn, project, slug)),
        'loop_node_path': LOOP_NODE,
        'assist_run_id': str(_assist_run(conn, project, slug)),
        'partition_id': str(_partition(conn, project)),
        'import_id': str(_import(conn, project, slug)),
        'source_id': str(source),
        'train_job_id': str(train_job),
        'promote_id': str(promote),
        'bakeoff_job_id': BAKEOFF_JOB.format(slug=slug),
        # the file name of an export's frozen registry the seeded export holds
        'export_artifact': 'data.yaml',
    }


def seed_world_files(
    record: Any, run_job_id: str, tmp_path: Path, loop_id: str | None = None
) -> None:
    """The run's log and one artifact under its prefix, the project's bake-off results, the
    current export's frozen files and the member blob of the seeded loop node."""
    from src.config import get_curation_config
    from src.config.project_context import bind_project
    from src.services.training.run_retention import bakeoff_out_dir

    slug = record.slug
    run_dir = tmp_path / 'projects' / slug / 'training' / run_job_id
    run_dir.mkdir(parents=True)
    (run_dir / 'run.log').write_text(f'{slug}-epoch 1\n{slug}-epoch 2\n', encoding='utf-8')
    (run_dir / 'results.csv').write_text(f'epoch\n1\n# {slug}\n', encoding='utf-8')

    with bind_project(record):
        if loop_id is not None:
            from src.services.curation.assist_loop.snapshots import SnapshotStore

            asyncio.run(
                SnapshotStore(slug, uuid.UUID(loop_id)).write_ids(
                    'nodes', LOOP_NODE, loop_node_ids(slug)
                )
            )
        cfg = get_curation_config()
        job_dir = bakeoff_out_dir(cfg.bakeoff_results_dir) / BAKEOFF_JOB.format(slug=slug)
        export_root = cfg.export_root
        export = export_root / f'{slug}-export'
    job_id = job_dir.name
    dataset_dir = job_dir / BAKEOFF_DATASET.replace(':', '__')
    dataset_dir.mkdir(parents=True)
    dataset = {'id': BAKEOFF_DATASET}
    (job_dir / 'status.json').write_text(
        json.dumps(
            {
                'schema_version': 2,
                'job_id': job_id,
                'state': 'done',
                'profile': None,
                'datasets': [BAKEOFF_DATASET],
                'models': [],
                'started_at': None,
                'finished_at': None,
                'progress': {'done': 0, 'total': 0},
                'completed': [],
                'failed': [],
                'error': None,
            }
        ),
        encoding='utf-8',
    )
    (dataset_dir / 'comparison.json').write_text(
        json.dumps(
            {
                'schema_version': 2,
                'job_id': job_id,
                'profile': None,
                'thresholds': {},
                'dataset': dataset,
                'eval_classes': [],
                'common_classes': [],
                'rank_by': 'map_50',
                'rank_scope': 'overall',
                'models': [],
                'failed': [],
                'warnings': [],
                'n_models': 0,
            }
        ),
        encoding='utf-8',
    )
    (job_dir / 'matrix.json').write_text(
        json.dumps(
            {
                'schema_version': 2,
                'job_id': job_id,
                'rank_by': None,
                'datasets': [],
                'models': [],
                'metrics': [],
                'cells': {},
                'best': {},
            }
        ),
        encoding='utf-8',
    )

    export.mkdir(parents=True)
    for name in ('class_registry.json', 'manifest.json', 'label_stats.json'):
        (export / name).write_text(json.dumps({'owner': slug}), encoding='utf-8')
    (export / 'data.yaml').write_text(f'names: [{slug}_class]\n', encoding='utf-8')
    (export_root / 'current').symlink_to(export.name)
