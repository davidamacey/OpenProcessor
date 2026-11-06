#!/usr/bin/env python3
"""Generate the committed 0.6.0 control-data fixture of ``tests/upgrade/fixtures/v060/``.

    python -m scripts.db.make_v060_fixture [--out DIR]

A small synthetic 0.6.0 install (public names and numbers only): three projects (``default``, in
the pre-folding layout with separate settings and view-state indexes and a ``class_registry.json``
file; ``demo``, folded, with the class registry as a document; ``lean``, folded, whose axes point at built-ins with no
stored revision), one deleted slug, classes with a
merge, hotkeys and the R-3 name pairs, prompt-pack revisions with an activation history, a VLM
endpoint with a probe, a settings document, a keymap, a holdout freeze, label operations, an
ingest source with a ``queue.sqlite``, a dataset import with a ledger, a training run, a
promotion, a combine run, an active reprocess job and the leftover auto-label files.

Index bodies are in ``v060_index_bodies.py``. The output is deterministic: running it twice gives the same bytes.
"""

from __future__ import annotations

import argparse
import base64
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from scripts.db.v060_index_bodies import (
    CLASSES_BODY,
    GLOBAL_CONFIGS_INDEX_BODY,
    LABELS_CONFIRMED_BODY,
    PROJECTS_INDEX_BODY,
    SETTINGS_BODY,
    UMAP_STATE_BODY,
    UMAP_VIZ_STATE_BODY,
    configs_body as _configs_body,
    images_body,
    index,
    items_body,
)


GENERIC_REGION_PACK_NAME = 'generic_region_v1'  # a built-in prompt pack that is not the default
ROOT = '{ROOT}'
PREFIX = 'op_prj_'
T0 = '2026-01-01T00:00:00+00:00'
T1 = '2026-01-02T00:00:00+00:00'
T2 = '2026-01-03T00:00:00+00:00'
OUT_DEFAULT = Path(__file__).resolve().parents[2] / 'tests' / 'upgrade' / 'fixtures' / 'v060'


def roles(slug: str, *, folded: bool) -> dict[str, str]:
    names = {
        r: f'{PREFIX}{slug}__{r}'
        for r in (
            'images',
            'items',
            'labels_confirmed',
            'classes',
            'settings',
            'umap_state',
            'umap_viz_state',
            'configs',
        )
    }
    if folded:
        names['settings'] = names['umap_viz_state'] = names['configs']
    return names


def project_doc(slug: str, name: str, *, folded: bool, status: str = 'active') -> dict[str, Any]:
    data, state = f'{ROOT}/data/projects/{slug}', f'{ROOT}/state/projects/{slug}'
    return {
        'slug': slug,
        'display_name': name,
        'description': f'{name} (synthetic)',
        'status': status,
        'revision': 3,
        'created_at': T0,
        'updated_at': T1,
        'origin': None,
        'pre_delete_status': None,
        'resources': {
            'indexes': roles(slug, folded=folded),
            'class_registry_path': f'{data}/class_registry.json',
            'export_root': f'{data}/exports',
            'upload_root': f'{state}/uploads',
            'bakeoff_eval_root': f'{data}/bakeoff_eval',
            'project_state_dir': state,
            'train_jobs_dir': f'{ROOT}/jobs/projects/{slug}',
            'autolabel_dir': f'{ROOT}/jobs/auto_label/projects/{slug}',
            'bakeoff_jobs_dir': f'{state}/bakeoff_jobs',
            'mlflow_experiment': f'openprocessor-{slug}',
            'model_prefix': '' if slug == 'default' else f'{slug}__',
        },
    }


def config(
    prefix: str, kind: str, name: str, revision: int, body: dict[str, Any]
) -> dict[str, Any]:
    cur = {
        'doc_type': 'config', 'kind': kind, 'name': name, 'revision': revision, 'body': body,
        'description': f'{name} {kind}', 'created_at': T0, 'updated_at': T1, 'updated_by': None,
        'cloned_from': None,
    }  # fmt: skip
    return {f'{prefix}:{name}': cur}


def revision_copy(
    prefix: str, kind: str, name: str, revision: int, body: dict[str, Any]
) -> dict[str, Any]:
    doc = config(prefix, kind, name, revision, body)[f'{prefix}:{name}']
    return {
        f'{prefix}:{name}@{revision}': {
            **doc,
            'doc_type': 'revision',
            'updated_at': f'2026-01-0{revision}T00:00:00+00:00',
        }
    }


def activation(
    axis: str, name: str | None, revision: int | None, previous: dict[str, Any] | None, at: str
) -> dict[str, Any]:
    return {
        f'activation:{axis}': {
            'doc_type': 'activation',
            'axis': axis,
            'name': name,
            'revision': revision,
            'activated_at': at,
            'previous': previous,
        }
    }


def event(
    n: int,
    axis: str,
    name: str | None,
    revision: int | None,
    previous: dict[str, Any] | None,
    at: str,
) -> dict[str, Any]:
    return {
        f'activation_event:00000000-0000-4000-8000-{n:012d}': {
            'doc_type': 'activation_event',
            'axis': axis,
            'name': name,
            'revision': revision,
            'previous': previous,
            'activated_at': at,
        }
    }


def configs_docs() -> dict[str, Any]:
    pack1 = {'prompts': {'car': 'a photo of a car'}, 'version': 1}
    pack2 = {'prompts': {'car': 'a photo of a car', 'bus': 'a photo of a bus'}, 'version': 2}
    profile = {'region_class_name': 'plate', 'detector': 'yolo26_small_trt'}
    docs: dict[str, Any] = {}
    docs.update(config('pack', 'prompt_pack', 'signs', 2, pack2))
    docs.update(revision_copy('pack', 'prompt_pack', 'signs', 1, pack1))
    docs.update(revision_copy('pack', 'prompt_pack', 'signs', 2, pack2))
    docs.update(config('profile', 'region_profile', 'cars', 1, profile))
    docs.update(revision_copy('profile', 'region_profile', 'cars', 1, profile))
    docs.update(
        config('ingest', 'ingest_profile', 'fast', 1, {'detector': 'yolov11_small_trt_end2end'})
    )
    docs.update(
        revision_copy(
            'ingest', 'ingest_profile', 'fast', 1, {'detector': 'yolov11_small_trt_end2end'}
        )
    )
    docs.update(activation('prompt_pack', 'signs', 2, {'name': 'signs', 'revision': 1}, T2))
    docs.update(activation('detection_profile', 'cars', 1, None, T1))
    docs.update(activation('vlm', 'local-vlm', 1, None, T1))
    docs.update(event(1, 'prompt_pack', 'signs', 1, None, T1))
    docs.update(event(2, 'prompt_pack', 'signs', 2, {'name': 'signs', 'revision': 1}, T2))
    docs.update(event(3, 'detection_profile', 'cars', 1, None, T1))
    docs.update(event(4, 'vlm', 'local-vlm', 1, None, T1))
    # an axis that was activated and then turned off: the event log ends in "off"
    docs.update(event(5, 'ingest_profile', 'fast', 1, None, T1))
    docs.update(event(6, 'ingest_profile', None, None, {'name': 'fast', 'revision': 1}, T2))
    docs.update(activation('ingest_profile', None, None, {'name': 'fast', 'revision': 1}, T2))
    docs['meta:config_revision'] = {'doc_type': 'meta', 'config_revision': 9}
    docs['keymap:default'] = {
        'doc_type': 'keymap',
        'overrides': {'next_item': ['j']},
        'revision': 3,
        'updated_at': T2,
    }
    docs['runtime:detection:host1'] = {
        'doc_type': 'runtime',
        'process': 'detection',
        'host': 'host1',
        'applied_at': T2,
    }
    return docs


def settings_doc() -> dict[str, Any]:
    return {
        'defaults': {'scorer': 'margin', 'clusterer': None},
        'ingest_policy': {'revision': 2, 'mode': 'all'},
        'vlm_policy': {'revision': 1, 'max_images': 4},
        'cluster_refresh': {
            'revision': 4,
            'last_trained_count': 120,
            'trained_before': None,
            'prev_count': 100,
            'job_id': None,
        },
        'updated_at': T2,
        'updated_by': None,
    }


def registry_entry(cid: int, name: str, **extra: Any) -> dict[str, Any]:
    return {'id': cid, 'name': name, 'group': extra.pop('group', 'unknown'), 'added': T0, **extra}


def classes_file() -> dict[str, Any]:
    return {
        'version': 1,
        'updated_at': T2,
        'classes': [
            registry_entry(0, 'car', group='vehicle', hotkey_letter='c'),
            registry_entry(1, 'Traffic Light', group='signal', deprecated=True, merged_into=2),
            registry_entry(
                2, 'traffic_light', group='signal', hotkey_letter='t', notes='merged target'
            ),
            registry_entry(3, 'Person', group='people'),
            registry_entry(4, 'bus', group='vehicle', deprecated=True),
            registry_entry(5, 'bus', group='vehicle'),
        ],
    }


def classes_doc() -> dict[str, Any]:
    return {'class_registry': {**classes_file(), 'next_class_id': 7, 'registry_revision': 12}}


def item(
    crop_id: str,
    class_id: int | None,
    name: str | None,
    *,
    vector: bool = True,
    state: str | None = 'embedded',
) -> dict[str, Any]:
    doc: dict[str, Any] = {'crop_id': crop_id, 'class_id': class_id, 'class_name': name}
    if vector:
        doc['pe_embedding'] = [0.1, 0.2]
    if state:
        doc['embedding_state'] = state
    return doc


OP_ENTRY = {
    'op_id': 'lop_20260105T101010000001_aaaaaa', 'writer': 'labeler', 'class_id': 3,
    'class_name': 'Person', 'class_source': 'detector', 'class_validated': False, 'restorable': True,
}  # fmt: skip
OTHER_ENTRY = {
    'op_id': 'lop_20251201T000000000000_ffffff',
    'writer': 'labeler',
    'class_id': 1,
    'restorable': True,
}


def sqlite_queue() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'queue.sqlite'
        conn = sqlite3.connect(path)
        conn.executescript(
            'CREATE TABLE objects (key TEXT NOT NULL, etag TEXT NOT NULL, version TEXT, size INTEGER,'
            ' state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, error TEXT,'
            ' queued_at REAL NOT NULL, updated_at REAL NOT NULL, PRIMARY KEY (key, etag));'
            'CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);'
        )
        rows = [
            ('images/a.jpg', 'e1', None, 10, 'pending', 0, None, 1767225600.0, 1767225600.0),
            ('images/b.jpg', 'e2', None, 11, 'pending', 1, 'timeout', 1767225601.0, 1767225602.0),
            ('images/c.jpg', 'e3', None, 12, 'done', 1, None, 1767225603.0, 1767225604.0),
            ('images/d.jpg', 'e4', None, 13, 'dead', 3, 'gone', 1767225605.0, 1767225606.0),
        ]
        conn.executemany('INSERT INTO objects VALUES (?,?,?,?,?,?,?,?,?)', rows)
        conn.commit()
        conn.execute('PRAGMA journal_mode=DELETE')
        conn.close()
        return base64.b64encode(path.read_bytes()).decode('ascii')


def jsonl(*rows: dict[str, Any]) -> str:
    return ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows)


def build_files() -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []

    def add(path: str, content: str | bytes, mtime: str = T1) -> None:
        if isinstance(content, bytes):
            files.append(
                {
                    'path': path,
                    'enc': 'b64',
                    'data': base64.b64encode(content).decode('ascii'),
                    'mtime': mtime,
                }
            )
        else:
            files.append({'path': path, 'enc': 'utf8', 'data': content, 'mtime': mtime})

    def js(path: str, value: Any, mtime: str = T1) -> None:
        add(path, json.dumps(value, indent=2, sort_keys=True), mtime)

    state = 'state/projects/demo'
    freeze = {
        'kind': 'curated', 'frozen_at': T1, 'test_holdout_sha': 'a' * 64, 'n_frozen': 3,
        'cohort_spec': {'min_per_class': 5}, 'per_class_counts': {'car': 3}, 'crop_ids': ['c1', 'c2', 'c3'],
    }  # fmt: skip
    add(f'{state}/pipeline_paused.flag', '', T2)
    js(f'{state}/test_holdout/current.json', freeze)
    js(f'{state}/test_holdout/20260102T000000000000Z.json', freeze)
    js(
        f'{state}/test_holdout/imports/20260103T000000000000Z.json',
        {
            **freeze,
            'kind': 'import',
            'test_holdout_sha': 'b' * 64,
            'n_frozen': 2,
            'crop_ids': ['c4', 'c5'],
        },
    )
    js(f'{state}/last_clustering.json', {'n_clusters': 12})
    fp = {
        'op_id': 'lop_20260105T101010000001_aaaaaa',
        'writer': 'labeler',
        'at': T2,
        'class_name': 'car',
        'class_source': 'human',
        'class_validated': True,
        'cluster_id': None,
    }
    for op, status, members in (
        ('lop_20260105T101010000001_aaaaaa', 'written', ['c1', 'c2', 'c3']),
        ('lop_20260105T101010000002_bbbbbb', 'writing', ['c9']),
    ):
        js(
            f'{state}/label_ops/{op}/state.json',
            {
                'schema_version': 1,
                'op_id': op,
                'kind': 'label',
                'writer': 'labeler',
                'at': T2,
                'class_name': 'car',
                'class_id': 0,
                'target': 'crop_ids',
                'selected': len(members),
                'count': len(members),
                'fingerprint': fp,
                'status': status,
                'undo_passes': [],
            },
        )
        js(f'{state}/label_ops/{op}/members.json', {'ids': members})
    add(
        'data/projects/default/class_registry.json',
        json.dumps(classes_file(), indent=2, sort_keys=True),
    )
    js(
        'data/projects/demo/exports/export_1/manifest.json',
        {'class_names': ['car', 'bus'], 'seed': 7, 'created_at': T2, 'dataset_sha': 'c' * 64},
    )
    run = 'train_20260104T000000_ab12cd34'
    base = 'jobs/projects/demo'
    js(
        f'{base}/{run}.job.json',
        {
            'job_id': run,
            'model_family': 'yolo26',
            'model_size': 's',
            'dataset_export_dir': f'{ROOT}/data/projects/demo/exports/export_1',
            'submitted_at': T2,
            'project': 'demo',
        },
    )
    js(
        f'{base}/{run}.status.json',
        {
            'job_id': run,
            'state': 'finished',
            'started_at': T2,
            'finished_at': '2026-01-04T01:00:00+00:00',
            'mlflow_run_id': 'mlf123',
        },
    )
    js(f'{base}/{run}.manifest.json', {'dataset_sha': 'c' * 64, 'map50': 0.5})
    js(
        f'{base}/{run}.registry_snapshot.json',
        {'classes': [{'id': 0, 'name': 'car'}, {'id': 1, 'name': 'bus'}]},
    )
    add(f'{base}/{run}.run.log', 'epoch 1/1 ok\n')
    add('jobs/auto_label/projects/demo/trigger.json', json.dumps({'requested_at': T2}))
    add('jobs/auto_label/projects/demo/running.lock', '12345')
    source = {
        'source_id': 'src_demo01',
        'name': 'Camera A',
        'backend': 'folder',
        'path': '/data/images/a',
        'extensions': ['.jpg'],
        'source_tag': 'camera_a',
        'batch_size': 32,
        'concurrency': 4,
        'watch': {'enabled': True, 'interval_s': 60},
        'schedule': {'enabled': False, 'cron': ''},
        'webhook_enabled': False,
        'created_at': T0,
        'updated_at': T1,
    }
    js('jobs/bulk_ingest/projects/demo/sources/src_demo01/source.json', source)
    js(
        'jobs/bulk_ingest/projects/demo/sources/src_demo01/state.json',
        {'last_poll_started_at': 1767225600, 'last_run_id': 'bulk_20260103T000000_aaaaaaaa'},
    )
    add(
        'jobs/bulk_ingest/projects/demo/sources/src_demo01/queue.sqlite',
        base64.b64decode(sqlite_queue()),
        T2,
    )
    js(
        'jobs/bulk_ingest/projects/demo/bulk_20260103T000000_aaaaaaaa/state.json',
        {'status': 'completed', 'started_at': T2, 'finished_at': T2},
    )
    imp = 'jobs/imports/projects/demo/imp_20260104T000000_c0ffee01'
    js(
        f'{imp}/request.json',
        {'options': {'name': 'coco subset'}, 'source': {'path': '/data/coco'}},
    )
    js(f'{imp}/mapping.json', {'classes': {'car': 'car'}})
    js(f'{imp}/scan.json', {'images_total': 3, 'chunks_total': 1})
    js(
        f'{imp}/state.json',
        {
            'import_id': 'imp_20260104T000000_c0ffee01',
            'status': 'completed',
            'name': 'coco subset',
            'source_format': 'coco',
            'source_root': '/data/coco',
            'started_at': T2,
            'finished_at': T2,
            'report': {'images_created': 2},
        },
    )
    add(
        f'{imp}/ledger/000000.jsonl',
        jsonl(
            {
                'rel_path': 'a.jpg',
                'status': 'pending',
                'image_id': 'img1',
                'items': [{'crop_id': 'c1', 'action': 'created'}],
            },
            {
                'rel_path': 'a.jpg',
                'status': 'ok',
                'image_id': 'img1',
                'items': [
                    {'crop_id': 'c1', 'action': 'created'},
                    {'crop_id': 'c2', 'action': 'standalone'},
                ],
            },
            {'rel_path': 'b.jpg', 'status': 'skipped', 'error_kind': 'image_size_mismatch'},
            {
                'rel_path': 'c.jpg',
                'status': 'failed',
                'error_kind': 'decode_failed',
                'error': 'bad jpeg',
            },
        ),
    )
    add(f'{imp}/chunks_done.jsonl', jsonl({'chunk': 0, 'report': {}}))
    js(
        'jobs/reprocess/projects/demo/rep_20260106T000000_11111111/state.json',
        {'status': 'running', 'started_at': T2, 'images_done': 5, 'images_total': 50},
    )
    add('jobs/reprocess/projects/demo/rep_20260106T000000_11111111/heartbeat', '')
    js(
        'jobs/reprocess/projects/demo/rep_20260105T000000_22222222/state.json',
        {'status': 'completed', 'started_at': T1, 'finished_at': T2},
    )
    js('jobs/region_drain/projects/demo/state.json', {'streak': 2, 'zero_since': None})
    js('jobs/region_drain/projects/demo/dependency_health.json', {'ok': True, 'checked_at': T2})
    cmb = 'jobs/combine/cmb_20260107T000000_deadbeef'
    js(
        f'{cmb}/state.json',
        {
            'status': 'completed',
            'target': 'demo',
            'sources': ['default'],
            'started_at': T2,
            'finished_at': T2,
        },
    )
    js(f'{cmb}/request.json', {'holdout': 'union', 'target': 'demo'})
    js(
        f'{cmb}/mapping.json',
        {
            'target_classes': ['car'],
            'per_source': {'default': {'car': {'kind': 'map', 'class_name': 'car'}}},
        },
    )
    js(
        'models/demo__car_detector/promote.json',
        {
            'job_id': run,
            'triton_name': 'demo__car_detector',
            'version': '1',
            'promoted_at': '2026-01-04T02:00:00+00:00',
            'project': 'demo',
            'classes': [{'model_id': 0, 'name': 'car'}, {'model_id': 1, 'name': 'bus'}],
            'class_remap_source': 'registry',
            'class_remap': {'source': 'registry'},
            'shared': True,
            'sharing_revision': 2,
        },
    )
    js('models/demo__car_detector/class_remap.json', {'0': 0, '1': 5})
    return files


def build_opensearch() -> dict[str, Any]:
    default_roles = roles('default', folded=False)
    demo_roles = roles('demo', folded=True)
    lean_roles = roles('lean', folded=True)
    out: dict[str, Any] = {}
    projects = {
        'project:default': project_doc('default', 'Default', folded=False),
        'project:demo': project_doc('demo', 'Demo', folded=True),
        'project:lean': project_doc('lean', 'Lean', folded=True),
        'project:old-demo': {
            **project_doc('old-demo', 'Old demo', folded=True, status='deleted'),
            'updated_at': T2,
        },
        'meta:projects_revision': {'revision': 5},
    }
    out['op_projects'] = index(PROJECTS_INDEX_BODY, projects)
    vlm_body = {'base_url': 'http://vlm:8000/v1', 'model': 'qwen3-vl-4b', 'auth': None}
    glob: dict[str, Any] = {}
    glob.update(config('vlm', 'vlm_endpoint', 'local-vlm', 1, vlm_body))
    glob.update(revision_copy('vlm', 'vlm_endpoint', 'local-vlm', 1, vlm_body))
    glob['vlm_probe:local-vlm@1'] = {
        'doc_type': 'vlm_probe',
        'name': 'local-vlm',
        'probe_key': 'local-vlm@1',
        'probed_at': T2,
        'body': {
            'fingerprint': 'f' * 16,
            'record': {
                'ok': True,
                'probed_at': T2,
                'latency_ms': 120.5,
                'json_mode_supported': True,
            },
        },
    }
    glob['vlm_probe:env'] = {
        'doc_type': 'vlm_probe',
        'name': 'env',
        'probe_key': 'env',
        'probed_at': T2,
        'body': {'record': {'ok': False, 'probed_at': T2}},
    }
    glob['local_vlm:desired'] = {
        'doc_type': 'local_vlm',
        'catalog_id': 'qwen3-vl-4b',
        'requested_at': T2,
    }
    glob['meta:config_revision'] = {'doc_type': 'meta', 'config_revision': 3}
    out['op_global_configs'] = index(GLOBAL_CONFIGS_INDEX_BODY, glob)
    # default: the pre-folding layout, eight separate indexes
    out[default_roles['images']] = index(images_body('faiss'))
    out[default_roles['items']] = index(items_body('faiss'), {
        'i1': item('i1', 0, 'car'), 'i2': item('i2', 2, 'Traffic Light'), 'i3': item('i3', 99, 'ghost'),
        'i4': item('i4', 0, 'truck'), 'i5': item('i5', 5, 'bus', state=None),
    })  # fmt: skip
    out[default_roles['configs']] = index(
        _configs_body(), {**configs_docs(), 'lease:label_undo': {'lease': {'holder': 'x'}}}
    )
    out[default_roles['settings']] = index(
        SETTINGS_BODY, {'default': {'defaults': {'scorer': 'entropy'}, 'updated_at': T1}}
    )
    out[default_roles['umap_viz_state']] = index(
        UMAP_VIZ_STATE_BODY,
        {
            'current': {
                'state_id': 'current',
                'projection_version': 'umap_viz_v1',
                'scope': 'all',
                'cluster_id': None,
                'n_points': 5,
                'fitted_at': T2,
                'n_components': 2,
                'metric': 'cosine',
            }
        },
    )
    out[default_roles['umap_state']] = index(
        UMAP_STATE_BODY,
        {
            'current': {
                'state_id': 'current',
                'reducer_b64': base64.b64encode(b'reducer-bytes').decode('ascii'),
                'n_components': 2,
                'metric': 'cosine',
            }
        },
    )
    out[default_roles['classes']] = index(
        CLASSES_BODY, {'0': {'class_id': 0, 'class_name': 'car', 'sample_count': 2}}
    )
    out[default_roles['labels_confirmed']] = index(
        LABELS_CONFIRMED_BODY, {'l1': {'label_id': 'l1', 'class_id': 0, 'class_name': 'car'}}
    )
    # demo: folded layout (settings and view state live in the configs index)
    out[demo_roles['images']] = index(images_body('lucene'))
    out[demo_roles['items']] = index(items_body('lucene'), {
        'd1': item('d1', 0, 'car'), 'd2': item('d2', 5, 'bus', state=None), 'd3': item('d3', None, None, vector=False, state=None),
        # members of the label operation: c1 recorded what the operation replaced, c2 was pruned, c3 is gone
        'c1': {**item('c1', 0, 'car'), 'class_id_history': [OTHER_ENTRY, OP_ENTRY]},
        'c2': {**item('c2', 0, 'car'), 'class_id_history': [OTHER_ENTRY]},
    })  # fmt: skip
    # demo moved from the stored endpoint to the environment built-in (no revision) and acknowledged both
    env_vlm = {
        **activation('vlm', 'env', None, {'name': 'local-vlm', 'revision': 1}, T2),
        **event(7, 'vlm', 'env', None, {'name': 'local-vlm', 'revision': 1}, T2),
    }
    env_vlm['activation:vlm']['acked_refs'] = {'local-vlm@1': T1, 'env@0123456789ab': T2}
    env_vlm['activation:vlm']['external_ack_at'] = T2
    folded = {**configs_docs(), **env_vlm, 'default': settings_doc(), 'class_registry': classes_doc(),
              'current': {'state_id': 'current', 'projection_version': 'umap_viz_v1', 'scope': 'all', 'cluster_id': None, 'n_points': 3, 'fitted_at': T2, 'n_components': 2, 'metric': 'cosine'}}  # fmt: skip
    out[demo_roles['configs']] = index(_configs_body(), folded)
    # lean activated built-ins that have no stored revision: a prompt pack and the env region profile
    lean = {
        **activation('prompt_pack', GENERIC_REGION_PACK_NAME, None, None, T1),
        **event(8, 'prompt_pack', GENERIC_REGION_PACK_NAME, None, None, T1),
        **activation('detection_profile', 'region', None, None, T2),
        **event(9, 'detection_profile', 'region', None, None, T2),
    }
    out[lean_roles['images']] = index(images_body('lucene'))
    out[lean_roles['items']] = index(items_body('lucene'))
    out[lean_roles['configs']] = index(_configs_body(), lean)
    out[lean_roles['umap_state']] = index(UMAP_STATE_BODY)
    out[lean_roles['classes']] = index(CLASSES_BODY)
    out[lean_roles['labels_confirmed']] = index(LABELS_CONFIRMED_BODY)
    out[demo_roles['umap_state']] = index(UMAP_STATE_BODY)
    out[demo_roles['classes']] = index(CLASSES_BODY)
    out[demo_roles['labels_confirmed']] = index(LABELS_CONFIRMED_BODY)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT_DEFAULT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'opensearch.json').write_text(
        json.dumps(build_opensearch(), indent=1, sort_keys=True) + '\n', encoding='utf-8'
    )
    (args.out / 'files.json').write_text(
        json.dumps(build_files(), indent=1, sort_keys=True) + '\n', encoding='utf-8'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
